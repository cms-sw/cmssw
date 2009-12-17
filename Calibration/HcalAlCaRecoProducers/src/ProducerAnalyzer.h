#ifndef CalibrationHcalAlCaRecoProducersProducerAnalyzer_h
#define CalibrationHcalAlCaRecoProducersProducerAnalyzer_h

// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace cms
{

//
// class declaration
//

class ProducerAnalyzer : public edm::EDAnalyzer {
   public:
      explicit ProducerAnalyzer(const edm::ParameterSet&);
      ~ProducerAnalyzer();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob() ;
      virtual void endJob() ;

   private:
  // ----------member data ---------------------------
  std::string nameProd_;
  std::string jetCalo_;
  std::string gammaClus_;
  std::string ecalInput_;
  std::string hbheInput_;
  std::string hoInput_;
  std::string hfInput_;
  std::string Tracks_;
};
}// end namespace cms
#endif
