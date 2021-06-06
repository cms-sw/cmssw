/*! \brief   Implementation of methods of TTClusterBuilder.h
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 *  \author Nicola
 *  \date   2013, Jul 12
 *
 */

#include "L1Trigger/TrackTrigger/plugins/TTClusterBuilder.h"

/// Implement the producer
template <>
void TTClusterBuilder<Ref_Phase2TrackerDigi_>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  /// Prepare output
  auto ttClusterDSVForOutput = std::make_unique<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>>();
  std::map<DetId, std::vector<Ref_Phase2TrackerDigi_>> rawHits;
  this->RetrieveRawHits(rawHits, iEvent);

  // Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle = iSetup.getHandle(tTopoToken);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  edm::ESHandle<TrackerGeometry> tGeomHandle = iSetup.getHandle(tGeomToken);
  const TrackerGeometry* const theTrackerGeom = tGeomHandle.product();

  // Loop on the OT stacks
  for (auto gd = theTrackerGeom->dets().begin(); gd != theTrackerGeom->dets().end(); gd++) {
    DetId detid = (*gd)->geographicalId();
    if (detid.subdetId() == 1 || detid.subdetId() == 2)
      continue;  // only run on OT
    if (!tTopo->isLower(detid))
      continue;  // loop on the stacks: choose the lower arbitrarily
    DetId lowerDetid = detid;
    DetId upperDetid = tTopo->partnerDetId(detid);

    /// Temp vectors containing the vectors of the
    /// hits used to build each cluster
    std::vector<std::vector<Ref_Phase2TrackerDigi_>> lowerHits, upperHits;

    /// Find the hits in each stack member
    typename std::map<DetId, std::vector<Ref_Phase2TrackerDigi_>>::const_iterator lowerHitFind =
        rawHits.find(lowerDetid);
    typename std::map<DetId, std::vector<Ref_Phase2TrackerDigi_>>::const_iterator upperHitFind =
        rawHits.find(upperDetid);

    /// If there are hits, cluster them
    /// It is the TTClusterAlgorithm::Cluster method which
    /// calls the constructor to the Cluster class!
    bool isPSP = (theTrackerGeom->getDetectorType(lowerDetid) == TrackerGeometry::ModuleType::Ph2PSP);
    if (lowerHitFind != rawHits.end())
      theClusterFindingAlgoHandle->Cluster(lowerHits, lowerHitFind->second, isPSP);
    if (upperHitFind != rawHits.end())
      theClusterFindingAlgoHandle->Cluster(upperHits, upperHitFind->second, false);

    /// Create TTCluster objects and store them
    /// Use the FastFiller with edmNew::DetSetVector
    {
      edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>::FastFiller lowerOutputFiller(*ttClusterDSVForOutput,
                                                                                            lowerDetid);
      for (unsigned int i = 0; i < lowerHits.size(); i++) {
        TTCluster<Ref_Phase2TrackerDigi_> temp(lowerHits.at(i), lowerDetid, 0, storeLocalCoord);
        lowerOutputFiller.push_back(temp);
      }
      if (lowerOutputFiller.empty())
        lowerOutputFiller.abort();
    }
    {
      edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>::FastFiller upperOutputFiller(*ttClusterDSVForOutput,
                                                                                            upperDetid);
      for (unsigned int i = 0; i < upperHits.size(); i++) {
        TTCluster<Ref_Phase2TrackerDigi_> temp(upperHits.at(i), upperDetid, 1, storeLocalCoord);
        upperOutputFiller.push_back(temp);
      }
      if (upperOutputFiller.empty())
        upperOutputFiller.abort();
    }
  }  /// End of loop over detector elements

  /// Put output in the event
  iEvent.put(std::move(ttClusterDSVForOutput), "ClusterInclusive");
}

/// Retrieve hits from the event
template <>
void TTClusterBuilder<Ref_Phase2TrackerDigi_>::RetrieveRawHits(
    std::map<DetId, std::vector<Ref_Phase2TrackerDigi_>>& mRawHits, const edm::Event& iEvent) {
  mRawHits.clear();
  /// Loop over the tags used to identify hits in the cfg file
  std::vector<edm::EDGetTokenT<edm::DetSetVector<Phase2TrackerDigi>>>::iterator it;
  for (it = rawHitTokens.begin(); it != rawHitTokens.end(); ++it) {
    /// For each tag, get the corresponding handle
    edm::Handle<edm::DetSetVector<Phase2TrackerDigi>> HitHandle;
    iEvent.getByToken(*it, HitHandle);
    edm::DetSetVector<Phase2TrackerDigi>::const_iterator detsIter;
    edm::DetSet<Phase2TrackerDigi>::const_iterator hitsIter;

    /// Loop over detector elements identifying Digis
    for (detsIter = HitHandle->begin(); detsIter != HitHandle->end(); detsIter++) {
      DetId id = detsIter->id;
      for (hitsIter = detsIter->data.begin(); hitsIter != detsIter->data.end(); hitsIter++) {
        mRawHits[id].push_back(edm::makeRefTo(HitHandle, id, hitsIter));
      }
    }
  }
}
