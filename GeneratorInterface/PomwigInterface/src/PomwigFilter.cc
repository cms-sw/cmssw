#include "GeneratorInterface/PomwigInterface/interface/PomwigFilter.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

PomwigFilter::PomwigFilter(const edm::ParameterSet& ppp) 
{}

PomwigFilter::~PomwigFilter() 
{}

bool
PomwigFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
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
PomwigFilter::beginJob(const edm::EventSetup&)
{
}

void 
PomwigFilter::endJob() {
}

//define this as a plug-in
