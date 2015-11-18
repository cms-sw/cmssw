#ifndef HINECALGENEVTSELECTOR_h
#define HINECALGENEVTSELECTOR_h


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


//
// class decleration
//
namespace edm {
  class HepMCProduct;
}

class HINEcalGenEvtSelector : public edm::EDFilter {
   public:
      explicit HINEcalGenEvtSelector(const edm::ParameterSet&);
      ~HINEcalGenEvtSelector();


      virtual bool filter(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

      edm::EDGetTokenT<edm::HepMCProduct> token_;
      std::vector<int> partonId_;
      std::vector<int> partonStatus_;
      std::vector<double> partonPt_;

      std::vector<int> particleId_;
      std::vector<int> particleStatus_;
      std::vector<double> particlePt_;

      double etaMax_;
};
#endif
