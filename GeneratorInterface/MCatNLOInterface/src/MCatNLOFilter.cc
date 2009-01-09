#include "GeneratorInterface/MCatNLOInterface/interface/MCatNLOFilter.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

MCatNLOFilter::MCatNLOFilter(const edm::ParameterSet& ppp) 
{}

MCatNLOFilter::~MCatNLOFilter() 
{}

bool
MCatNLOFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
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
MCatNLOFilter::beginJob(const edm::EventSetup&)
{
}

void 
MCatNLOFilter::endJob() {
}

//define this as a plug-in
