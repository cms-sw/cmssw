#ifndef __PFClusterTimeAssigner__
#define __PFClusterTimeAssigner__

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "DataFormats/Common/interface/ValueMap.h"


class PFClusterTimeAssigner : public edm::stream::EDProducer<> {
public: 
  PFClusterTimeAssigner(const edm::ParameterSet& conf) {
    const edm::InputTag&  clusters = 
      conf.getParameter<edm::InputTag>("src");    
    clustersTok_ = consumes<reco::PFClusterCollection>( clusters );

    const edm::InputTag& times = 
      conf.getParameter<edm::InputTag>("timeSrc");
    timesTok_ = consumes<edm::ValueMap<float> >( times );

    const edm::InputTag& timeResos = 
      conf.getParameter<edm::InputTag>("timeResoSrc");
    timeResosTok_ = consumes<edm::ValueMap<float> >( timeResos );
    
    produces<reco::PFClusterCollection>();
  }

  void produce(edm::Event& e, const edm::EventSetup& es) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:  
  edm::EDGetTokenT<reco::PFClusterCollection> clustersTok_;
  edm::EDGetTokenT<edm::ValueMap<float> > timesTok_;
  edm::EDGetTokenT<edm::ValueMap<float> > timeResosTok_;
};

DEFINE_FWK_MODULE(PFClusterTimeAssigner);

void PFClusterTimeAssigner::
produce(edm::Event& e, const edm::EventSetup& es) {
  auto clusters_out = std::make_unique<reco::PFClusterCollection>();
  
  edm::Handle<reco::PFClusterCollection> clustersH;
  e.getByToken(clustersTok_,clustersH);
  edm::Handle<edm::ValueMap<float> > timesH, timeResosH;
  e.getByToken(timesTok_,timesH);
  e.getByToken(timeResosTok_,timeResosH);

  auto const & clusters = *clustersH;
  auto const & times = *timesH;
  auto const & timeResos = *timeResosH;
  
  clusters_out->reserve(clusters.size());
  clusters_out->insert(clusters_out->end(),
		       clusters.begin(),clusters.end());

  //build the EE->PS association
  auto& out = *clusters_out;
  for( unsigned i = 0; i < out.size(); ++i ) {      
    
    edm::Ref<reco::PFClusterCollection> clusterRef(clustersH,i);
    const float time = times[clusterRef];
    const float timeReso = timeResos[clusterRef];
    out[i].setTime(time, timeReso);
    
  }

  e.put(std::move(clusters_out));
}

void PFClusterTimeAssigner::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;  
  desc.add<edm::InputTag>("src",edm::InputTag("particleFlowClusterECALUncorrected"));
  desc.add<edm::InputTag>("timeSrc",edm::InputTag("ecalBarrelClusterFastTimer"));
  desc.add<edm::InputTag>("timeResoSrc",edm::InputTag("ecalBarrelClusterFastTimer"));
  descriptions.add("particleFlowClusterTimeAssignerDefault",desc);
}

#endif
