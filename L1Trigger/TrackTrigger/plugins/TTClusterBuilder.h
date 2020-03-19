/*! \class   TTClusterBuilder
 *  \brief   Plugin to load the Clustering algorithm and produce the
 *           collection of Clusters that goes in the event content.
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 12
 *
 */

#ifndef L1_TRACK_TRIGGER_CLUSTER_BUILDER_H
#define L1_TRACK_TRIGGER_CLUSTER_BUILDER_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm.h"
#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithmRecord.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"

#include <memory>
#include <map>
#include <vector>

template <typename T>
class TTClusterBuilder : public edm::EDProducer {
  /// NOTE since pattern hit correlation must be performed within a stacked module, one must store
  /// Clusters in a proper way, providing easy access to them in a detector/member-wise way
public:
  /// Constructors
  explicit TTClusterBuilder(const edm::ParameterSet& iConfig);
  /// Destructor
  ~TTClusterBuilder() override;

private:
  /// Data members
  edm::ESHandle<TTClusterAlgorithm<T> > theClusterFindingAlgoHandle;
  std::vector<edm::EDGetTokenT<edm::DetSetVector<Phase2TrackerDigi> > > rawHitTokens;
  unsigned int ADCThreshold;
  bool storeLocalCoord;

  /// Mandatory methods
  void beginRun(const edm::Run& run, const edm::EventSetup& iSetup) override;
  void endRun(const edm::Run& run, const edm::EventSetup& iSetup) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  /// Get hits
  void RetrieveRawHits(std::map<DetId, std::vector<T> >& mRawHits, const edm::Event& iEvent);

};  /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Constructors
template <typename T>
TTClusterBuilder<T>::TTClusterBuilder(const edm::ParameterSet& iConfig) {
  ADCThreshold = iConfig.getParameter<unsigned int>("ADCThreshold");
  storeLocalCoord = iConfig.getParameter<bool>("storeLocalCoord");

  std::vector<edm::InputTag> rawHitInputTags = iConfig.getParameter<std::vector<edm::InputTag> >("rawHits");
  for (auto it = rawHitInputTags.begin(); it != rawHitInputTags.end(); ++it) {
    rawHitTokens.push_back(consumes<edm::DetSetVector<Phase2TrackerDigi> >(*it));
  }

  produces<edmNew::DetSetVector<TTCluster<T> > >("ClusterInclusive");
}

/// Destructor
template <typename T>
TTClusterBuilder<T>::~TTClusterBuilder() {}

/// Begin run
template <typename T>
void TTClusterBuilder<T>::beginRun(const edm::Run& run, const edm::EventSetup& iSetup) {
  /// Get the clustering algorithm
  iSetup.get<TTClusterAlgorithmRecord>().get(theClusterFindingAlgoHandle);
}

/// End run
template <typename T>
void TTClusterBuilder<T>::endRun(const edm::Run& run, const edm::EventSetup& iSetup) {}

/// Implement the producer
template <>
void TTClusterBuilder<Ref_Phase2TrackerDigi_>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup);

/// Retrieve hits from the event
template <>
void TTClusterBuilder<Ref_Phase2TrackerDigi_>::RetrieveRawHits(
    std::map<DetId, std::vector<Ref_Phase2TrackerDigi_> >& mRawHits, const edm::Event& iEvent);

#endif
