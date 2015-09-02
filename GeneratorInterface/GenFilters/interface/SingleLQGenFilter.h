//
//
//  Created by D. Nash, copied from LQGenFilter
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

//
// class declaration
//

class SingleLQGenFilter : public edm::EDFilter {
   public:
      explicit SingleLQGenFilter(const edm::ParameterSet&);
      ~SingleLQGenFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      edm::InputTag src_;
      bool eej_, enuej_, nuenuej_, mumuj_, munumuj_, numunumuj_; 
};
