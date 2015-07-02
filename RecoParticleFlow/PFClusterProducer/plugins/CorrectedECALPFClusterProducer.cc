#ifndef __CorrectedECALPFClusterProducer__
#define __CorrectedECALPFClusterProducer__

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

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEMEnergyCorrector.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "TVector2.h"

namespace {
  typedef reco::PFCluster::EEtoPSAssociation::value_type EEPSPair;
  bool sortByKey(const EEPSPair& a, const EEPSPair& b) {
    return a.first < b.first;
  } 
  double testPreshowerDistance(const edm::Ptr<reco::PFCluster>& eeclus,
			       const edm::Ptr<reco::PFCluster>& psclus) {
    if( psclus.isNull() ) return -1.0;
    /* 
    // commented out since PFCluster::layer() uses a lot of CPU
    // and since 
    if( PFLayer::ECAL_ENDCAP != eeclus->layer() ) return -1.0;
    if( PFLayer::PS1 != psclus->layer() &&
	PFLayer::PS2 != psclus->layer()    ) {
      throw cms::Exception("testPreshowerDistance")
	<< "The second argument passed to this function was "
	<< "not a preshower cluster!" << std::endl;
    } 
    */
    const reco::PFCluster::REPPoint& pspos = psclus->positionREP();
    const reco::PFCluster::REPPoint& eepos = eeclus->positionREP();
    // lazy continue based on geometry
    if( eeclus->z()*psclus->z() < 0 ) return -1.0;
    const double dphi= std::abs(TVector2::Phi_mpi_pi(eepos.phi() - 
						     pspos.phi()));
    if( dphi > 0.6 ) return -1.0;    
    const double deta= std::abs(eepos.eta() - pspos.eta());    
    if( deta > 0.3 ) return -1.0; 
    return LinkByRecHit::testECALAndPSByRecHit(*eeclus,*psclus,false);
  }
}

class CorrectedECALPFClusterProducer : public edm::stream::EDProducer<> {
public: 
  CorrectedECALPFClusterProducer(const edm::ParameterSet& conf):
    _minimumPSEnergy(conf.getParameter<double>("minimumPSEnergy")) {
    const edm::InputTag&  inputECAL = 
      conf.getParameter<edm::InputTag>("inputECAL");    
    _inputECAL = consumes<reco::PFClusterCollection>( inputECAL );

    const edm::InputTag& inputPS = 
      conf.getParameter<edm::InputTag>("inputPS");
    _inputPS = consumes<reco::PFClusterCollection>( inputPS );

    const edm::ParameterSet corConf = conf.getParameterSet("energyCorrector");
    _corrector.reset(new PFClusterEMEnergyCorrector(corConf,consumesCollector()));

    produces<reco::PFCluster::EEtoPSAssociation>();
    produces<reco::PFClusterCollection>();
  }

  virtual void produce(edm::Event& e, const edm::EventSetup& es);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const double _minimumPSEnergy;
  std::unique_ptr<PFClusterEMEnergyCorrector> _corrector;
  edm::EDGetTokenT<reco::PFClusterCollection> _inputECAL;
  edm::EDGetTokenT<reco::PFClusterCollection> _inputPS;
};

DEFINE_FWK_MODULE(CorrectedECALPFClusterProducer);

void CorrectedECALPFClusterProducer::
produce(edm::Event& e, const edm::EventSetup& es) {
  std::auto_ptr<reco::PFClusterCollection> clusters_out;
  clusters_out.reset(new reco::PFClusterCollection);    
  std::auto_ptr<reco::PFCluster::EEtoPSAssociation> association_out;
  association_out.reset(new reco::PFCluster::EEtoPSAssociation);
  
  edm::Handle<reco::PFClusterCollection> handleECAL;
  e.getByToken(_inputECAL,handleECAL);
  edm::Handle<reco::PFClusterCollection> handlePS;
  e.getByToken(_inputPS,handlePS);
  
  clusters_out->reserve(handleECAL->size());
  association_out->reserve(handleECAL->size());
  clusters_out->insert(clusters_out->end(),
		       handleECAL->begin(),handleECAL->end());
  //build the EE->PS association
  double dist = -1.0, min_dist = -1.0;
  for( unsigned i = 0; i < handlePS->size(); ++i ) {      
    switch( handlePS->at(i).layer() ) { // just in case this isn't the ES...
    case PFLayer::PS1:
    case PFLayer::PS2:
      break;
    default:
      continue;
    }    
    edm::Ptr<reco::PFCluster> psclus(handlePS,i);
    if( psclus->energy() < _minimumPSEnergy ) continue;
    edm::Ptr<reco::PFCluster> eematch,eeclus;
    dist = min_dist = -1.0; // reset
    for( size_t ic = 0; ic < handleECAL->size(); ++ic ) {
      if( handleECAL->at(ic).layer() != PFLayer::ECAL_ENDCAP ) continue;
      eeclus = edm::Ptr<reco::PFCluster>(handleECAL,ic);	
      dist = testPreshowerDistance(eeclus,psclus);      
      if( dist == -1.0 || (min_dist != -1.0 && dist > min_dist) ) continue;
      if( dist < min_dist || min_dist == -1.0 ) {
	eematch = eeclus;
	min_dist = dist;
      }
    } // loop on EE clusters      
    if( eematch.isNonnull() ) {      
      association_out->push_back(std::make_pair(eematch.key(),psclus));
    }
  }
  std::sort(association_out->begin(),association_out->end(),sortByKey);
  
  _corrector->correctEnergies(e,es,*association_out,*clusters_out);
  
  association_out->shrink_to_fit();
  
  e.put(association_out);
  e.put(clusters_out);
}

void CorrectedECALPFClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("minimumPSEnergy",0.0);
  desc.add<edm::InputTag>("inputPS",edm::InputTag("particleFlowClusterPS"));
  {
    edm::ParameterSetDescription psd0;
    psd0.add<bool>("applyCrackCorrections",false);
    psd0.add<bool>("applyMVACorrections",false);
    psd0.add<double>("maxPtForMVAEvaluation",-99.);
    psd0.add<std::string>("algoName","PFClusterEMEnergyCorrector");
    psd0.add<edm::InputTag>("recHitsEBLabel",edm::InputTag("ecalRecHit","EcalRecHitsEB"));
    psd0.add<edm::InputTag>("recHitsEELabel",edm::InputTag("ecalRecHit","EcalRecHitsEE"));
    psd0.add<edm::InputTag>("verticesLabel",edm::InputTag("offlinePrimaryVertices"));
    psd0.add<bool>("autoDetectBunchSpacing",true);
    psd0.add<int>("bunchSpacing",25);
    desc.add<edm::ParameterSetDescription>("energyCorrector",psd0);
  }
  desc.add<edm::InputTag>("inputECAL",edm::InputTag("particleFlowClusterECALUncorrected"));
  descriptions.add("particleFlowClusterECAL",desc);
}

#endif
