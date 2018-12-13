// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

//
// class declaration
//

class Herwig6Filter : public edm::EDFilter {
   public:
      explicit Herwig6Filter(const edm::ParameterSet&);
      ~Herwig6Filter();

   private:
      virtual bool filter(edm::Event&, const edm::EventSetup&);
};

Herwig6Filter::Herwig6Filter(const edm::ParameterSet& ppp) 
{}

Herwig6Filter::~Herwig6Filter() 
{}

bool
Herwig6Filter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   std::vector< Handle<HepMCProduct> > AllProds;
   iEvent.getManyByType(AllProds);
   
   if(AllProds.size()==0) {
     LogInfo("")<<"   Event is skipped and removed.\n";
     return false;
   }
   else return true;
}

//define this as a plug-in

DEFINE_FWK_MODULE(Herwig6Filter);
