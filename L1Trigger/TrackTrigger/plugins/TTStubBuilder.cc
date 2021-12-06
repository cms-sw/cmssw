/*!  \brief   Implementation of methods of TTClusterBuilder.h
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 * \author Andrew W. Rose
 * \author Nicola Pozzobon
 * \author Ivan Reid
 * \author Ian Tomalin
 * \date 2013 - 2020
 *
 */

#include "L1Trigger/TrackTrigger/plugins/TTStubBuilder.h"

/// Update output stubs with Refs to cluster collection that is associated to stubs.

template <>
void TTStubBuilder<Ref_Phase2TrackerDigi_>::updateStubs(
    const edm::OrphanHandle<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>>& clusterHandle,
    const edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>& inputEDstubs,
    edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>& outputEDstubs) const {
  /// Loop over tracker modules
  for (const auto& module : inputEDstubs) {
    /// Get the DetId and prepare the FastFiller
    DetId thisStackedDetId = module.id();
    typename edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>::FastFiller outputFiller(outputEDstubs,
                                                                                           thisStackedDetId);

    /// detid of the two components.
    ///This should be done via a TrackerTopology method that is not yet available.
    DetId lowerDetid = thisStackedDetId + 1;
    DetId upperDetid = thisStackedDetId + 2;

    /// Get the DetSets of the clusters
    edmNew::DetSet<TTCluster<Ref_Phase2TrackerDigi_>> lowerClusters = (*clusterHandle)[lowerDetid];
    edmNew::DetSet<TTCluster<Ref_Phase2TrackerDigi_>> upperClusters = (*clusterHandle)[upperDetid];

    /// Get the DetSet of the stubs
    edmNew::DetSet<TTStub<Ref_Phase2TrackerDigi_>> theseStubs = inputEDstubs[thisStackedDetId];

    /// Prepare the new DetSet to replace the current one
    /// Loop over the stubs in this module
    typename edmNew::DetSet<TTCluster<Ref_Phase2TrackerDigi_>>::const_iterator clusterIter;
    for (const auto& stub : theseStubs) {
      /// Create a temporary stub
      TTStub<Ref_Phase2TrackerDigi_> tempTTStub(stub.getDetId());

      /// Compare the clusters stored in the stub with the ones of this module
      const edm::Ref<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>, TTCluster<Ref_Phase2TrackerDigi_>>&
          lowerClusterToBeReplaced = stub.clusterRef(0);
      const edm::Ref<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>, TTCluster<Ref_Phase2TrackerDigi_>>&
          upperClusterToBeReplaced = stub.clusterRef(1);

      bool lowerOK = false;
      bool upperOK = false;

      for (clusterIter = lowerClusters.begin(); clusterIter != lowerClusters.end() && !lowerOK; ++clusterIter) {
        if (clusterIter->getHits() == lowerClusterToBeReplaced->getHits()) {
          tempTTStub.addClusterRef(edmNew::makeRefTo(clusterHandle, clusterIter));
          lowerOK = true;
        }
      }

      for (clusterIter = upperClusters.begin(); clusterIter != upperClusters.end() && !upperOK; ++clusterIter) {
        if (clusterIter->getHits() == upperClusterToBeReplaced->getHits()) {
          tempTTStub.addClusterRef(edmNew::makeRefTo(clusterHandle, clusterIter));
          upperOK = true;
        }
      }

      /// If no compatible clusters were found, skip to the next one
      if (!lowerOK || !upperOK)
        continue;

      /// getters for RawBend & BendOffset are in FULL-strip units, setters are in HALF-strip units
      tempTTStub.setRawBend(2. * stub.rawBend());
      tempTTStub.setBendOffset(2. * stub.bendOffset());
      tempTTStub.setBendBE(stub.bendBE());
      tempTTStub.setModuleTypePS(stub.moduleTypePS());

      outputFiller.push_back(tempTTStub);

    }  /// End of loop over stubs of this module
  }    /// End of loop over stub DetSetVector
}

/// Implement the producer

template <>
void TTStubBuilder<Ref_Phase2TrackerDigi_>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle = iSetup.getHandle(tTopoToken);
  const TrackerTopology* const tTopo = tTopoHandle.product();
  edm::ESHandle<TrackerGeometry> tGeomHandle = iSetup.getHandle(tGeomToken);
  const TrackerGeometry* const theTrackerGeom = tGeomHandle.product();

  /// -- Prepare output
  /// TTClusters associated to TTStubs
  auto ttClusterDSVForOutputAcc = std::make_unique<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>>();
  auto ttClusterDSVForOutputRej = std::make_unique<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>>();
  /// TTStubs with Refs to clusters pointing to collection of clusters in entire tracker.
  auto ttStubDSVForOutputAccTemp = std::make_unique<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>();
  auto ttStubDSVForOutputRejTemp = std::make_unique<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>();
  /// TTStubs with Refs to clusters pointing to collection of clusters associated to stubs.
  auto ttStubDSVForOutputAcc = std::make_unique<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>();
  auto ttStubDSVForOutputRej = std::make_unique<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>();

  /// Read the Clusters from the entire tracker.
  edm::Handle<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>> clusterHandle;
  iEvent.getByToken(clustersToken, clusterHandle);

  int nmod = -1;

  // Loop over all the tracker elements

  for (const auto& gd : theTrackerGeom->dets()) {
    DetId detid = (*gd).geographicalId();
    if (detid.subdetId() != StripSubdetector::TOB && detid.subdetId() != StripSubdetector::TID)
      continue;  // only run on OT
    if (!tTopo->isLower(detid))
      continue;  // loop on the stacks: choose the lower sensor
    DetId lowerDetid = detid;
    DetId upperDetid = tTopo->partnerDetId(detid);
    DetId stackDetid = tTopo->stack(detid);
    bool isPS = (theTrackerGeom->getDetectorType(stackDetid) == TrackerGeometry::ModuleType::Ph2PSP);

    bool is10G_PS = false;

    // Determine if this module is a 10G transmission scheme module
    //
    // TO FIX: take this info from Tracker -> DTC cabling map.

    if (detid.subdetId() == StripSubdetector::TOB) {
      if (tTopo->layer(detid) <= high_rate_max_layer)
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

    /// If there are Clusters in both sensors, you can try and make a Stub
    /// This is ~redundant
    if (lowerClusters.empty() || upperClusters.empty())
      continue;

    /// Create the vectors of objects to be passed to the FastFillers
    std::vector<TTCluster<Ref_Phase2TrackerDigi_>> tempClusLowerAcc;
    std::vector<TTCluster<Ref_Phase2TrackerDigi_>> tempClusLowerRej;
    std::vector<TTCluster<Ref_Phase2TrackerDigi_>> tempClusUpperAcc;
    std::vector<TTCluster<Ref_Phase2TrackerDigi_>> tempClusUpperRej;
    std::vector<TTStub<Ref_Phase2TrackerDigi_>> tempStubAcc;
    std::vector<TTStub<Ref_Phase2TrackerDigi_>> tempStubRej;
    tempClusLowerAcc.clear();
    tempClusLowerRej.clear();
    tempClusUpperAcc.clear();
    tempClusUpperRej.clear();
    tempStubAcc.clear();
    tempStubRej.clear();

    /// Get chip size information
    /// FIX ME: Should take this from TrackerTopology, but it was buggy in 2017 (SV)
    const int chipSize = isPS ? 120 : 127;
    // No. of macro pixels along local y in each half of 2S module.
    constexpr int numMacroPixels = 16;

    /// Loop over pairs of Clusters
    for (auto lowerClusterIter = lowerClusters.begin(); lowerClusterIter != lowerClusters.end(); ++lowerClusterIter) {
      /// Temporary storage to allow only one stub per inner cluster, if requested in cfi
      std::vector<TTStub<Ref_Phase2TrackerDigi_>> tempOutput;

      for (auto upperClusterIter = upperClusters.begin(); upperClusterIter != upperClusters.end(); ++upperClusterIter) {
        /// Build a temporary Stub
        TTStub<Ref_Phase2TrackerDigi_> tempTTStub(stackDetid);
        tempTTStub.addClusterRef(edmNew::makeRefTo(clusterHandle, lowerClusterIter));
        tempTTStub.addClusterRef(edmNew::makeRefTo(clusterHandle, upperClusterIter));
        tempTTStub.setModuleTypePS(isPS);

        /// Check for compatibility of cluster pair
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
      }    /// End of loop over upper clusters

      /// Here tempOutput stores all the stubs from this lower cluster
      /// Check if there is need to store only one or two (2S/PS modules cases) (if only one already, skip this step)
      if (ForbidMultipleStubs && tempOutput.size() > 1 + static_cast<unsigned int>(isPS)) {
        /// If so, sort the stubs by bend and keep only the first one (2S case) or the first pair (PS case) (smallest |bend|)
        std::sort(tempOutput.begin(), tempOutput.end(), TTStubBuilder<Ref_Phase2TrackerDigi_>::SortStubsBend);

        /// Get to the second element (the switch above ensures there are min 2)
        typename std::vector<TTStub<Ref_Phase2TrackerDigi_>>::iterator tempIter = tempOutput.begin();
        ++tempIter;
        if (isPS)
          ++tempIter;  // PS module case
        /// tempIter points now to the second or third element (2S/PS)

        /// Delete all-but-the first one from tempOutput
        tempOutput.erase(tempIter, tempOutput.end());
      }

      /// Here, tempOutput is either of size 1 (if ForbidMultupleStubs = true),
      /// or of size N with all the valid combinations ...

      /// Now loop over the accepted stubs (1 or N) for this lower cluster

      for (auto& tempTTStub : tempOutput) {
        /// Put in the output
        if (not applyFE)  // No dynamic inefficiencies
        {
          /// This means that ALL stubs go into the output
          tempClusLowerAcc.push_back(*(tempTTStub.clusterRef(0)));
          tempClusUpperAcc.push_back(*(tempTTStub.clusterRef(1)));
          tempStubAcc.push_back(tempTTStub);
        } else {
          bool FEreject = false;

          /// This means that only some stubs go to the output
          MeasurementPoint mp0 = tempTTStub.clusterRef(0)->findAverageLocalCoordinates();
          int seg = static_cast<int>(mp0.y());  // Identifies which half of module
          if (isPS)
            seg = seg / numMacroPixels;
          /// Find out which MPA/CBC ASIC
          int chip = 1000 * nmod + 10 * int(tempTTStub.innerClusterPosition() / chipSize) + seg;
          /// Find out which CIC ASIC
          int CIC_chip = 10 * nmod + seg;

          // First look is the stub is passing trough the very front end (CBC/MPA)
          maxStubs = isPS ? maxStubs_PS : maxStubs_2S;

          if (isPS)  // MPA
          {
            if (moduleStubs_MPA[chip] < int(maxStubs)) {
              moduleStubs_MPA[chip]++;
            } else {
              FEreject = true;
            }
          } else  // CBC
          {
            if (moduleStubs_CBC[chip] < int(maxStubs)) {
              moduleStubs_CBC[chip]++;
            } else {
              FEreject = true;
            }
          }

          // End of the MPA/CBC loop

          // If the stub has been already thrown out, there is no reason to include it into the CIC stream
          // We put it in the rejected container, flagged with offset to indicate reason.

          if (FEreject) {
            tempTTStub.setRawBend(CBCFailOffset + 2. * tempTTStub.rawBend());
            tempTTStub.setBendOffset(CBCFailOffset + 2. * tempTTStub.bendOffset());
            tempClusLowerRej.push_back(*(tempTTStub.clusterRef(0)));
            tempClusUpperRej.push_back(*(tempTTStub.clusterRef(1)));
            tempStubRej.push_back(tempTTStub);
            continue;
          }

          maxStubs = isPS ? maxStubs_PS_CIC_5 : maxStubs_2S_CIC_5;

          if (is10G_PS)
            maxStubs = maxStubs_PS_CIC_10;

          bool CIC_reject = true;

          moduleStubs_CIC[CIC_chip].push_back(tempTTStub);  //We temporarily add the new stub

          if (moduleStubs_CIC[CIC_chip].size() <= maxStubs) {
            tempClusLowerAcc.push_back(*(tempTTStub.clusterRef(0)));
            tempClusUpperAcc.push_back(*(tempTTStub.clusterRef(1)));
            tempStubAcc.push_back(tempTTStub);  // The stub is kept

          } else {
            /// TO FIX: The maxStub stubs with lowest |bend| are retained. This algo considers all stubs
            /// since the last multiple of 8 events, (i.e. in 1-8 events), whereas true CIC chip considers
            /// all stubs in current block of 8 events (i.e. always in 8 events). So may be optimistic.

            /// Sort stubs by |bend|.
            bendMap.clear();
            bendMap.reserve(moduleStubs_CIC[CIC_chip].size());

            for (unsigned int i = 0; i < moduleStubs_CIC[CIC_chip].size(); ++i) {
              bendMap.emplace_back(i, moduleStubs_CIC[CIC_chip].at(i).bendFE());
            }

            std::sort(bendMap.begin(), bendMap.end(), TTStubBuilder<Ref_Phase2TrackerDigi_>::SortStubBendPairs);

            // bendMap contains link over all the stubs included in moduleStubs_CIC[CIC_chip]

            for (unsigned int i = 0; i < maxStubs; ++i) {
              // The stub we have just added is amongst those with smallest |bend| so keep it.
              if (bendMap[i].first == moduleStubs_CIC[CIC_chip].size() - 1) {
                CIC_reject = false;
              }
            }

            if (CIC_reject)  // The stub added does not pass the cut
            {
              tempTTStub.setRawBend(CICFailOffset + 2. * tempTTStub.rawBend());
              tempTTStub.setBendOffset(CICFailOffset + 2. * tempTTStub.bendOffset());
              tempClusLowerRej.push_back(*(tempTTStub.clusterRef(0)));
              tempClusUpperRej.push_back(*(tempTTStub.clusterRef(1)));
              tempStubRej.push_back(tempTTStub);
            } else {
              tempClusLowerAcc.push_back(*(tempTTStub.clusterRef(0)));
              tempClusUpperAcc.push_back(*(tempTTStub.clusterRef(1)));
              tempStubAcc.push_back(tempTTStub);  // The stub is added
            }
          }
        }  /// End of check on max number of stubs per module
      }    /// End of nested loop
    }      /// End of loop over pairs of Clusters

    /// Fill output collections
    if (not tempClusLowerAcc.empty())
      this->fill(*ttClusterDSVForOutputAcc, lowerDetid, tempClusLowerAcc);
    if (not tempClusLowerRej.empty())
      this->fill(*ttClusterDSVForOutputRej, lowerDetid, tempClusLowerRej);
    if (not tempClusUpperAcc.empty())
      this->fill(*ttClusterDSVForOutputAcc, upperDetid, tempClusUpperAcc);
    if (not tempClusUpperRej.empty())
      this->fill(*ttClusterDSVForOutputRej, upperDetid, tempClusUpperRej);
    if (not tempStubAcc.empty())
      this->fill(*ttStubDSVForOutputAccTemp, stackDetid, tempStubAcc);
    if (not tempStubRej.empty())
      this->fill(*ttStubDSVForOutputRejTemp, stackDetid, tempStubRej);

  }  /// End of loop over detector elements

  // Store the subset of clusters associated to stubs.

  edm::OrphanHandle<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>> ttClusterAccHandle =
      iEvent.put(std::move(ttClusterDSVForOutputAcc), "ClusterAccepted");
  edm::OrphanHandle<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>> ttClusterRejHandle =
      iEvent.put(std::move(ttClusterDSVForOutputRej), "ClusterRejected");

  /// The TTStubs created above contain Refs to the ClusterInclusive collection (all clusters in Tracker).
  /// Replace them with new TTStubs with Refs to ClusterAccepted collection (only clusters in Stubs),
  /// so big ClusterInclusive collection can be dropped.

  this->updateStubs(ttClusterAccHandle, *ttStubDSVForOutputAccTemp, *ttStubDSVForOutputAcc);
  this->updateStubs(ttClusterRejHandle, *ttStubDSVForOutputRejTemp, *ttStubDSVForOutputRej);

  /// Put output in the event (2)
  iEvent.put(std::move(ttStubDSVForOutputAcc), "StubAccepted");
  iEvent.put(std::move(ttStubDSVForOutputRej), "StubRejected");

  ++ievt;
  if (ievt % 8 == 0)
    moduleStubs_CIC.clear();  // Everything is cleared up after 8BX
  if (ievt % 2 == 0)
    moduleStubs_MPA.clear();  // Everything is cleared up after 2BX
  moduleStubs_CBC.clear();    // Everything is cleared up after everyBX
}
