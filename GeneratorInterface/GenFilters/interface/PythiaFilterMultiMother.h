#ifndef PYTHIAFILTERMULTIMOTHER_h
#define PYTHIAFILTERMULTIMOTHER_h


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

class PythiaFilterMultiMother : public edm::EDFilter {
   public:
      explicit PythiaFilterMultiMother(const edm::ParameterSet&);
      ~PythiaFilterMultiMother() override;


      bool filter(edm::Event&, const edm::EventSetup&) override;
   private:
     // ----------member data ---------------------------

       edm::EDGetTokenT<edm::HepMCProduct> token_;
       int particleID;
       double minpcut;
       double maxpcut;
       double minptcut;
       double maxptcut;
       double minetacut;
       double maxetacut;
       double minrapcut;
       double maxrapcut;
       double minphicut;
       double maxphicut;

       double rapidity;

       int status;
       std::vector<int> motherIDs;
       int processID;

       double betaBoost;
};
#endif
