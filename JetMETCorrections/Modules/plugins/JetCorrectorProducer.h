#ifndef JetMETCorrections_Modules_JetCorrectorProducer_h
#define JetMETCorrections_Modules_JetCorrectorProducer_h
// -*- C++ -*-
//
// Package:     JetMETCorrections/Modules
// Class  :     JetCorrectorProducer
// 
/**\class JetCorrectorProducer JetCorrectorProducer.h "JetCorrectorProducer.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Sun, 31 Aug 2014 20:40:18 GMT
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

// forward declarations

template<typename T>
class JetCorrectorProducer : public edm::stream::EDProducer<>
{
 public:
 JetCorrectorProducer(edm::ParameterSet const& iPSet):
  maker_{iPSet,consumesCollector()}
  {
    produces<reco::JetCorrector>();
  }
  
  // ---------- member functions ---------------------------
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override {
    auto impl =maker_.make(iEvent,iSetup);
    std::auto_ptr<reco::JetCorrector> corrector{ new reco::JetCorrector{std::move(impl) } };
    iEvent.put(corrector);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& iDescriptions) {
    T::Maker::fillDescriptions(iDescriptions);
  }
  
 private:
  JetCorrectorProducer(const JetCorrectorProducer&) = delete; // stop default
  
  const JetCorrectorProducer& operator=(const JetCorrectorProducer&) = delete; // stop default
  
  // ---------- member data --------------------------------
  typename T::Maker maker_; 
};


#endif
