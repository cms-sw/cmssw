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
     edm::InputTag hbheLabel_;
     edm::InputTag hoLabel_;
     edm::InputTag hfLabel_;
     std::vector<edm::InputTag> ecalLabels_;
     std::vector<edm::InputTag> mInputCalo;
     std::string correctedIslandBarrelSuperClusterCollection_;
     std::string correctedIslandBarrelSuperClusterProducer_;
     std::string correctedIslandEndcapSuperClusterCollection_;
     std::string correctedIslandEndcapSuperClusterProducer_;
     
     bool allowMissingInputs_;
     std::string m_inputTrackLabel;
 // Calo geometry
  const CaloGeometry* geo;
  
};
//}// end namespace cms
#endif
