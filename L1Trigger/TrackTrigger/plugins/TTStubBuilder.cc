/*!  \brief   Implementation of methods of TTClusterBuilder.h
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 * \author Andrew W. Rose
 * \author Nicola Pozzobon
 * \author Ivan Reid
 * \date 2013, Jul 18
 *
 */

#include "L1Trigger/TrackTrigger/plugins/TTStubBuilder.h"

/// Implement the producer
template <>
void TTStubBuilder<Ref_Phase2TrackerDigi_>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
  edm::ESHandle<TrackerGeometry> tGeomHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get(tGeomHandle);
  const TrackerGeometry* const theTrackerGeom = tGeomHandle.product();

  /// Prepare output
  auto ttClusterDSVForOutput = std::make_unique<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>>();
  auto ttStubDSVForOutputTemp = std::make_unique<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>();
  auto ttStubDSVForOutputAccepted = std::make_unique<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>();
  auto ttStubDSVForOutputRejected = std::make_unique<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>();

  static constexpr int CBCFailOffset = 500;
  static constexpr int CICFailOffset = 1000;

  /// Get the Clusters already stored away
  edm::Handle<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>> clusterHandle;
  iEvent.getByToken(clustersToken, clusterHandle);

  int nmod = -1;

  // Loop over all the tracker elements

  //  for (auto gd=theTrackerGeom->dets().begin(); gd != theTrackerGeom->dets().end(); gd++)
  for (const auto& gd : theTrackerGeom->dets()) {
    DetId detid = (*gd).geographicalId();
    if (detid.subdetId() != StripSubdetector::TOB && detid.subdetId() != StripSubdetector::TID)
      continue;  // only run on OT
    if (!tTopo->isLower(detid))
      continue;  // loop on the stacks: choose the lower arbitrarily
    DetId lowerDetid = detid;
    DetId upperDetid = tTopo->partnerDetId(detid);
    DetId stackDetid = tTopo->stack(detid);
    bool isPS = (theTrackerGeom->getDetectorType(stackDetid) == TrackerGeometry::ModuleType::Ph2PSP);

    bool is10G_PS = false;

    // Determine if this module is a 10G transmission scheme module
    //
    // sviret comment (221217): this info should be made available in conddb at some point
    // not in TrackerTopology as some modules may switch between 10G and 5G transmission
    // schemes during running period

    if (detid.subdetId() == StripSubdetector::TOB) {
      if (tTopo->layer(detid) == 1)
        is10G_PS = true;
    } else if (detid.subdetId() == StripSubdetector::TID) {
      if (tTopo->tidRing(detid) <= high_rate_max_ring[tTopo->tidWheel(detid) - 1])
        is10G_PS = true;
    }

    ++nmod;

    unsigned int maxStubs;
    std::vector<std::pair<unsigned int, double>> bendMap;

    /// Go on only if both detectors have Clusters
    if (clusterHandle->find(lowerDetid) == clusterHandle->end() ||
        clusterHandle->find(upperDetid) == clusterHandle->end())
      continue;

    /// Get the DetSets of the Clusters
    edmNew::DetSet<TTCluster<Ref_Phase2TrackerDigi_>> lowerClusters = (*clusterHandle)[lowerDetid];
    edmNew::DetSet<TTCluster<Ref_Phase2TrackerDigi_>> upperClusters = (*clusterHandle)[upperDetid];

    /// If there are Clusters in both sensors
    /// you can try and make a Stub
    /// This is ~redundant
    if (lowerClusters.empty() || upperClusters.empty())
      continue;

    /// Create the vectors of objects to be passed to the FastFillers
    std::vector<TTCluster<Ref_Phase2TrackerDigi_>> tempInner;
    std::vector<TTCluster<Ref_Phase2TrackerDigi_>> tempOuter;
    std::vector<TTStub<Ref_Phase2TrackerDigi_>> tempAccepted;
    tempInner.clear();
    tempOuter.clear();
    tempAccepted.clear();

    /// Get chip size information
    int chipSize = 127;  /// SV 21/11/17: tracker topology method should be updated, currently provide wrong nums
    if (isPS)
      chipSize = 120;

    /// Loop over pairs of Clusters
    for (auto lowerClusterIter = lowerClusters.begin(); lowerClusterIter != lowerClusters.end(); ++lowerClusterIter) {
      /// Temporary storage to allow only one stub per inner cluster
      /// if requested in cfi
      std::vector<TTStub<Ref_Phase2TrackerDigi_>> tempOutput;

      for (auto upperClusterIter = upperClusters.begin(); upperClusterIter != upperClusters.end(); ++upperClusterIter) {
        /// Build a temporary Stub
        TTStub<Ref_Phase2TrackerDigi_> tempTTStub(stackDetid);
        tempTTStub.addClusterRef(edmNew::makeRefTo(clusterHandle, lowerClusterIter));
        tempTTStub.addClusterRef(edmNew::makeRefTo(clusterHandle, upperClusterIter));
        tempTTStub.setModuleTypePS(isPS);

        /// Check for compatibility
        bool thisConfirmation = false;
        int thisDisplacement = 999999;
        int thisOffset = 0;
        float thisHardBend = 0;

        theStubFindingAlgoHandle->PatternHitCorrelation(
            thisConfirmation, thisDisplacement, thisOffset, thisHardBend, tempTTStub);
        // Removed real offset.  Ivan Reid 10/2019

        /// If the Stub is above threshold
        if (thisConfirmation) {
          tempTTStub.setRawBend(thisDisplacement);
          tempTTStub.setBendOffset(thisOffset);
          tempTTStub.setBendBE(thisHardBend);
          tempOutput.push_back(tempTTStub);
        }  /// Stub accepted
      }    /// End of loop over outer clusters

      /// Here tempOutput stores all the stubs from this inner cluster
      /// Check if there is need to store only one (if only one already, skip this step)
      if (ForbidMultipleStubs && tempOutput.size() > 1) {
        /// If so, sort the stubs by bend and keep only the first one (smallest bend)
        std::sort(tempOutput.begin(), tempOutput.end(), TTStubBuilder<Ref_Phase2TrackerDigi_>::SortStubsBend);

        /// Get to the second element (the switch above ensures there are min 2)
        typename std::vector<TTStub<Ref_Phase2TrackerDigi_>>::iterator tempIter = tempOutput.begin();
        ++tempIter;

        /// tempIter points now to the second element

        /// Delete all-but-the first one from tempOutput
        tempOutput.erase(tempIter, tempOutput.end());
      }

      /// Here, tempOutput is either of size 1 (if entering the switch)
      /// either of size N with all the valid combinations ...

      /// Now loop over the accepted stubs (1 or N) for this inner cluster
      for (unsigned int iTempStub = 0; iTempStub < tempOutput.size(); ++iTempStub) {
        /// Get the stub
        const TTStub<Ref_Phase2TrackerDigi_>& tempTTStub = tempOutput[iTempStub];

        // A temporary stub, for FE problems
        TTStub<Ref_Phase2TrackerDigi_> tempTTStub2(tempTTStub.getDetId());

        tempTTStub2.addClusterRef((tempTTStub.clusterRef(0)));
        tempTTStub2.addClusterRef((tempTTStub.clusterRef(1)));
        tempTTStub2.setRawBend(2. * tempTTStub.rawBend());
        tempTTStub2.setBendOffset(2. * tempTTStub.bendOffset());
        tempTTStub2.setBendBE(tempTTStub.bendBE());
        tempTTStub2.setModuleTypePS(tempTTStub.moduleTypePS());

        /// Put in the output
        if (!applyFE)  // No dynamic inefficiencies (DEFAULT)
        {
          /// This means that ALL stubs go into the output
          tempInner.push_back(*(tempTTStub.clusterRef(0)));
          tempOuter.push_back(*(tempTTStub.clusterRef(1)));
          tempAccepted.push_back(tempTTStub);
        } else {
          bool FEreject = false;
          /// This means that only some of them do
          /// Put in the temporary output
          MeasurementPoint mp0 = tempTTStub.clusterRef(0)->findAverageLocalCoordinates();
          int seg = static_cast<int>(mp0.y());
          if (isPS)
            seg = seg / 16;
          int chip = 1000 * nmod + 10 * int(tempTTStub.innerClusterPosition() / chipSize) +
                     seg;                  /// Find out which MPA/CBC ASIC
          int CIC_chip = 10 * nmod + seg;  /// Find out which CIC ASIC

          // First look is the stub is passing trough the very front end (CBC/MPA)
          (isPS) ? maxStubs = maxStubs_PS : maxStubs = maxStubs_2S;

          if (isPS)  // MPA
          {
            if (moduleStubs_MPA.find(chip) == moduleStubs_MPA.end())  /// Already a stub for this ASIC?
            {
              /// No, so new entry
              moduleStubs_MPA.emplace(chip, 1);
            } else {
              if (moduleStubs_MPA[chip] < int(maxStubs)) {
                ++moduleStubs_MPA[chip];
              } else {
                FEreject = true;
              }
            }
          } else  // CBC
          {
            if (moduleStubs_CBC.find(chip) == moduleStubs_CBC.end())  /// Already a stub for this ASIC?
            {
              /// No, so new entry
              moduleStubs_CBC.emplace(chip, 1);
            } else {
              if (moduleStubs_CBC[chip] < int(maxStubs)) {
                ++moduleStubs_CBC[chip];
              } else {
                FEreject = true;
              }
            }
          }

          // End of the MPA/CBC loop

          // If the stub has been already thrown out, there is no reason to include it into the CIC stream
          // We keep is in the stub final container tough, but flagged as reject by FE

          if (FEreject) {
            tempTTStub2.setRawBend(CBCFailOffset + 2. * tempTTStub.rawBend());
            tempTTStub2.setBendOffset(CBCFailOffset + 2. * tempTTStub.bendOffset());

            tempInner.push_back(*(tempTTStub2.clusterRef(0)));
            tempOuter.push_back(*(tempTTStub2.clusterRef(1)));
            tempAccepted.push_back(tempTTStub2);
            continue;
          }

          maxStubs = isPS ? maxStubs_PS_CIC_5 : maxStubs_2S_CIC_5;

          if (is10G_PS)
            maxStubs = maxStubs_PS_CIC_10;

          if (moduleStubs_CIC.find(CIC_chip) == moduleStubs_CIC.end())  /// Already a stub for this ASIC?
          {
            if (moduleStubs_MPA.find(chip) == moduleStubs_MPA.end())  /// Already a stub for this ASIC?
            {
              /// No, so new entry
              moduleStubs_MPA.emplace(chip, 1);
            } else {
              if (moduleStubs_MPA[chip] < int(maxStubs)) {
                ++moduleStubs_MPA[chip];
              } else {
                FEreject = true;
              }
            }
          } else  // CBC
          {
            if (moduleStubs_CBC.find(chip) == moduleStubs_CBC.end())  /// Already a stub for this ASIC?
            {
              /// No, so new entry
              moduleStubs_CBC.emplace(chip, 1);
            } else {
              if (moduleStubs_CBC[chip] < int(maxStubs)) {
                ++moduleStubs_CBC[chip];
              } else {
                FEreject = true;
              }
            }
          }

          // End of the MPA/CBC loop

          // If the stub has been already thrown out, there is no reason to include it into the CIC stream
          // We keep is in the stub final container tough, but flagged as reject by FE

          if (FEreject) {
            tempTTStub2.setRawBend(CBCFailOffset + 2. * tempTTStub.rawBend());
            tempTTStub2.setBendOffset(CBCFailOffset + 2. * tempTTStub.bendOffset());

            tempInner.push_back(*(tempTTStub2.clusterRef(0)));
            tempOuter.push_back(*(tempTTStub2.clusterRef(1)));
            tempAccepted.push_back(tempTTStub2);
            continue;
          }

          (isPS) ? maxStubs = maxStubs_PS_CIC_5 : maxStubs = maxStubs_2S_CIC_5;

          if (is10G_PS)
            maxStubs = maxStubs_PS_CIC_10;

          if (moduleStubs_CIC.find(CIC_chip) == moduleStubs_CIC.end())  /// Already a stub for this ASIC?
          {
            bool CIC_reject = true;

            if (moduleStubs_CIC[CIC_chip].size() < maxStubs) {
              moduleStubs_CIC[CIC_chip].push_back(tempTTStub);  //We add the new stub
              tempInner.push_back(*(tempTTStub.clusterRef(0)));
              tempOuter.push_back(*(tempTTStub.clusterRef(1)));
              tempAccepted.push_back(tempTTStub);  // The stub is added
            } else {
              moduleStubs_CIC[CIC_chip].push_back(tempTTStub);  //We add the new stub

              /// Sort them by |bend| and pick up only the first N.
              bendMap.clear();
              bendMap.reserve(moduleStubs_CIC[CIC_chip].size());

              for (unsigned int i = 0; i < moduleStubs_CIC[CIC_chip].size(); ++i) {
                bendMap.emplace_back(i, moduleStubs_CIC[CIC_chip].at(i).bendFE());
              }

              std::sort(bendMap.begin(), bendMap.end(), TTStubBuilder<Ref_Phase2TrackerDigi_>::SortStubBendPairs);

              // bendMap contains link over all the stubs included in moduleStubs_CIC[CIC_chip]

              for (unsigned int i = 0; i < maxStubs; ++i) {
                // The stub we have added is among the first ones, add it
                if (bendMap[i].first == moduleStubs_CIC[CIC_chip].size() - 1) {
                  CIC_reject = false;
                }
              }

              if (CIC_reject)  // The stub added does not pass the cut
              {
                tempTTStub2.setRawBend(CICFailOffset + 2. * tempTTStub.rawBend());
                tempTTStub2.setBendOffset(CICFailOffset + 2. * tempTTStub.bendOffset());

                tempInner.push_back(*(tempTTStub2.clusterRef(0)));
                tempOuter.push_back(*(tempTTStub2.clusterRef(1)));
                tempAccepted.push_back(tempTTStub2);
              } else {
                tempInner.push_back(*(tempTTStub.clusterRef(0)));
                tempOuter.push_back(*(tempTTStub.clusterRef(1)));
                tempAccepted.push_back(tempTTStub);  // The stub is added
              }
            }
          }
        }  /// End of check on max number of stubs per module
      }    /// End of nested loop
    }      /// End of loop over pairs of Clusters

    /// Create the FastFillers
    if (!tempInner.empty()) {
      typename edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>::FastFiller lowerOutputFiller(
          *ttClusterDSVForOutput, lowerDetid);
      for (unsigned int m = 0; m < tempInner.size(); m++) {
        lowerOutputFiller.push_back(tempInner.at(m));
      }
      if (lowerOutputFiller.empty())
        lowerOutputFiller.abort();
    }

    if (!tempOuter.empty()) {
      typename edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>::FastFiller upperOutputFiller(
          *ttClusterDSVForOutput, upperDetid);
      for (unsigned int m = 0; m < tempOuter.size(); m++) {
        upperOutputFiller.push_back(tempOuter.at(m));
      }
      if (upperOutputFiller.empty())
        upperOutputFiller.abort();
    }

    if (!tempAccepted.empty()) {
      typename edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>::FastFiller tempAcceptedFiller(
          *ttStubDSVForOutputTemp, stackDetid);
      for (unsigned int m = 0; m < tempAccepted.size(); m++) {
        tempAcceptedFiller.push_back(tempAccepted.at(m));
      }
      if (tempAcceptedFiller.empty())
        tempAcceptedFiller.abort();
    }

  }  /// End of loop over detector elements

  /// Put output in the event (1)
  /// Get also the OrphanHandle of the accepted clusters
  edm::OrphanHandle<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>> ttClusterAcceptedHandle =
      iEvent.put(std::move(ttClusterDSVForOutput), "ClusterAccepted");

  /// Now, correctly reset the output
  typename edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>::const_iterator stubDetIter;

  for (stubDetIter = ttStubDSVForOutputTemp->begin(); stubDetIter != ttStubDSVForOutputTemp->end(); ++stubDetIter) {
    /// Get the DetId and prepare the FastFiller
    DetId thisStackedDetId = stubDetIter->id();
    typename edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>::FastFiller acceptedOutputFiller(
        *ttStubDSVForOutputAccepted, thisStackedDetId);

    /// detid of the two components.
    ///This should be done via a TrackerTopology method that is not yet available.
    DetId lowerDetid = thisStackedDetId + 1;
    DetId upperDetid = thisStackedDetId + 2;

    /// Get the DetSets of the clusters
    edmNew::DetSet<TTCluster<Ref_Phase2TrackerDigi_>> lowerClusters = (*ttClusterAcceptedHandle)[lowerDetid];
    edmNew::DetSet<TTCluster<Ref_Phase2TrackerDigi_>> upperClusters = (*ttClusterAcceptedHandle)[upperDetid];

    /// Get the DetSet of the stubs
    edmNew::DetSet<TTStub<Ref_Phase2TrackerDigi_>> theseStubs = (*ttStubDSVForOutputTemp)[thisStackedDetId];

    /// Prepare the new DetSet to replace the current one
    /// Loop over the stubs
    typename edmNew::DetSet<TTCluster<Ref_Phase2TrackerDigi_>>::iterator clusterIter;
    typename edmNew::DetSet<TTStub<Ref_Phase2TrackerDigi_>>::iterator stubIter;
    for (stubIter = theseStubs.begin(); stubIter != theseStubs.end(); ++stubIter) {
      /// Create a temporary stub
      TTStub<Ref_Phase2TrackerDigi_> tempTTStub(stubIter->getDetId());

      /// Compare the clusters stored in the stub with the ones of this module
      const edm::Ref<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>, TTCluster<Ref_Phase2TrackerDigi_>>&
          lowerClusterToBeReplaced = stubIter->clusterRef(0);
      const edm::Ref<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>, TTCluster<Ref_Phase2TrackerDigi_>>&
          upperClusterToBeReplaced = stubIter->clusterRef(1);

      bool lowerOK = false;
      bool upperOK = false;

      for (clusterIter = lowerClusters.begin(); clusterIter != lowerClusters.end() && !lowerOK; ++clusterIter) {
        if (clusterIter->getHits() == lowerClusterToBeReplaced->getHits()) {
          tempTTStub.addClusterRef(edmNew::makeRefTo(ttClusterAcceptedHandle, clusterIter));
          lowerOK = true;
        }
      }

      for (clusterIter = upperClusters.begin(); clusterIter != upperClusters.end() && !upperOK; ++clusterIter) {
        if (clusterIter->getHits() == upperClusterToBeReplaced->getHits()) {
          tempTTStub.addClusterRef(edmNew::makeRefTo(ttClusterAcceptedHandle, clusterIter));
          upperOK = true;
        }
      }

      /// If no compatible clusters were found, skip to the next one
      if (!lowerOK || !upperOK)
        continue;

      tempTTStub.setRawBend(2. * stubIter->rawBend());  /// getter is in FULL-strip units, setter is in HALF-strip units
      tempTTStub.setBendOffset(
          2. * stubIter->bendOffset());  /// getter is in FULL-strip units, setter is in HALF-strip units
      tempTTStub.setBendBE(stubIter->bendBE());
      tempTTStub.setModuleTypePS(stubIter->moduleTypePS());

      acceptedOutputFiller.push_back(tempTTStub);

    }  /// End of loop over stubs of this module

    if (acceptedOutputFiller.empty())
      acceptedOutputFiller.abort();

  }  /// End of loop over stub DetSetVector

  /// Put output in the event (2)
  iEvent.put(std::move(ttStubDSVForOutputAccepted), "StubAccepted");
  iEvent.put(std::move(ttStubDSVForOutputRejected), "StubRejected");

  ++ievt;
  if (ievt % 8 == 0)
    moduleStubs_CIC.clear();  // Everything is cleared up after 8BX
  if (ievt % 2 == 0)
    moduleStubs_MPA.clear();  // Everything is cleared up after 2BX
  moduleStubs_CBC.clear();    // Everything is cleared up after everyBX
}
