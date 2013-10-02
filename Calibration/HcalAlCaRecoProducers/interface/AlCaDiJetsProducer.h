#ifndef AlCaDiJetsProducer_h
#define AlCaDiJetsProducer_h


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

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

//
// class declaration
//
namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace cms
{

class AlCaDiJetsProducer : public edm::EDProducer {
   public:
     explicit AlCaDiJetsProducer(const edm::ParameterSet&);
     ~AlCaDiJetsProducer();

     virtual void beginJob() ;

     virtual void produce(edm::Event &, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
     std::vector<edm::InputTag> ecalLabels_;

     edm::EDGetTokenT<reco::CaloJetCollection> tok_jets_;
     edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
     edm::EDGetTokenT<HORecHitCollection> tok_ho_;
     edm::EDGetTokenT<HFRecHitCollection> tok_hf_;

     std::vector<edm::EDGetTokenT<EcalRecHitCollection> > toks_ecal_;

     bool allowMissingInputs_;

};
}// end namespace cms
#endif
