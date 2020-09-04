//---------------------------------------------------------------------------
// class VectorHitBuilderEDProducer
// author: ebrondol,nathera
// date: May, 2015
//---------------------------------------------------------------------------

#ifndef RecoLocalTracker_SiPhase2VectorHitBuilder_VectorHitBuilderEDProducer_h
#define RecoLocalTracker_SiPhase2VectorHitBuilder_VectorHitBuilderEDProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/SiPhase2VectorHitBuilder/interface/VectorHitBuilderAlgorithmBase.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "RecoLocalTracker/SiPhase2VectorHitBuilder/interface/VectorHitBuilderAlgorithm.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"
#include "RecoLocalTracker/Records/interface/TkPhase2OTCPERecord.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

class VectorHitBuilderEDProducer : public edm::stream::EDProducer<> {
public:
  explicit VectorHitBuilderEDProducer(const edm::ParameterSet&);
  ~VectorHitBuilderEDProducer() override;
  void produce(edm::Event&,  const edm::EventSetup&) override;
  void run(edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D> > clusters,
           edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersAcc,
           edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersRej,
           VectorHitCollectionNew& outputAcc,
           VectorHitCollectionNew& outputRej);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  const VectorHitBuilderAlgorithm* algo() const { return stubsBuilder_; };

private:
  const VectorHitBuilderAlgorithm * stubsBuilder_;
  std::string offlinestubsTag_;
  unsigned int maxOfflinestubs_;
  edm::ESGetToken<VectorHitBuilderAlgorithm, TkPhase2OTCPERecord> stubsBuilderToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<Phase2TrackerCluster1D> > clusterProducer_;

};

#endif
