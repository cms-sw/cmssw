#ifndef AlCaGammaJetProducer_AlCaHcalProducers_h
#define AlCaGammaJetProducer_AlCaHcalProducers_h


// -*- C++ -*-


// system include files
#include <memory>
#include <string>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

//
// class declaration
//
namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

//namespace cms
//{

class AlCaGammaJetProducer : public edm::EDProducer {
   public:
      explicit AlCaGammaJetProducer(const edm::ParameterSet&);
      ~AlCaGammaJetProducer();
      virtual void beginJob() ;

      virtual void produce(edm::Event &, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
     std::vector<edm::InputTag> ecalLabels_;
     std::vector<edm::InputTag> mInputCalo;

     std::vector<edm::EDGetTokenT<EcalRecHitCollection> > toks_ecal_;
     std::vector<edm::EDGetTokenT<reco::CaloJetCollection> > toks_calo_;

     std::string correctedIslandBarrelSuperClusterCollection_;
     std::string correctedIslandBarrelSuperClusterProducer_;
     std::string correctedIslandEndcapSuperClusterCollection_;
     std::string correctedIslandEndcapSuperClusterProducer_;

     edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
     edm::EDGetTokenT<HORecHitCollection> tok_ho_;
     edm::EDGetTokenT<HFRecHitCollection> tok_hf_;
     edm::EDGetTokenT<reco::TrackCollection> tok_inputTrack_;
     edm::EDGetTokenT<reco::SuperClusterCollection> tok_EBSC_; // token for corrected island barrel super cluster collection
      edm::EDGetTokenT<reco::SuperClusterCollection> tok_EESC_; // token for corrected island end-cap super cluster
      
     bool allowMissingInputs_;

 // Calo geometry
  const CaloGeometry* geo;
  
};
//}// end namespace cms
#endif
