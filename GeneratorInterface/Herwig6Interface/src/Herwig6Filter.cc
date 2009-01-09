#include "GeneratorInterface/Herwig6Interface/interface/Herwig6Filter.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


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


void 
Herwig6Filter::beginJob(const edm::EventSetup&)
{
}

void 
Herwig6Filter::endJob() {
}

//define this as a plug-in
