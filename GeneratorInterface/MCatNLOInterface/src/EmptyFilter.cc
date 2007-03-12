#include "GeneratorInterface/MCatNLOInterface/interface/EmptyFilter.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

EmptyFilter::EmptyFilter(const edm::ParameterSet& ppp) 
{}

EmptyFilter::~EmptyFilter() 
{}

bool
EmptyFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   std::vector< Handle<HepMCProduct> > AllProds;
   iEvent.getManyByType(AllProds);
   
   if(AllProds.size()==0) {
     std::cout<<"   Event is skipped and removed." << std::endl;
     return false;
   }
   else return true;
}


void 
EmptyFilter::beginJob(const edm::EventSetup&)
{
}

void 
EmptyFilter::endJob() {
}

//define this as a plug-in
