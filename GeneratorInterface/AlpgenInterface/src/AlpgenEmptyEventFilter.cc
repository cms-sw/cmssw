#include "GeneratorInterface/AlpgenInterface/interface/AlpgenEmptyEventFilter.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

AlpgenEmptyEventFilter::AlpgenEmptyEventFilter(const edm::ParameterSet& ppp) 
{}

AlpgenEmptyEventFilter::~AlpgenEmptyEventFilter() 
{}

bool
AlpgenEmptyEventFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
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
AlpgenEmptyEventFilter::beginJob(const edm::EventSetup&)
{
}

void 
AlpgenEmptyEventFilter::endJob() {
}

//define this as a plug-in
