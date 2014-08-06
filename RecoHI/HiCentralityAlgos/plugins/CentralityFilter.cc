//
// Original Author:  Yetkin Yilmaz,32 4-A08,+41227673039,
//         Created:  Tue Jun 29 12:19:49 CEST 2010
//
//


// system include files
#include <memory>
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoHI/HiCentralityAlgos/interface/CentralityProvider.h"

//
// class declaration
//

class CentralityFilter : public edm::EDFilter {
   public:
      explicit CentralityFilter(const edm::ParameterSet&);
      ~CentralityFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
   CentralityProvider * centrality_;
  std::vector<int> selectedBins_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CentralityFilter::CentralityFilter(const edm::ParameterSet& iConfig) :
  centrality_(0),
  selectedBins_(iConfig.getParameter<std::vector<int> >("selectedBins"))
{
   //now do what ever initialization is needed

}


CentralityFilter::~CentralityFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
CentralityFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  bool result = false;

   using namespace edm;
   if(!centrality_) centrality_ = new CentralityProvider(iSetup);
   centrality_->newEvent(iEvent,iSetup);
   int bin = centrality_->getBin();

   for(unsigned int i = 0; i < selectedBins_.size(); ++i){
     if(bin == selectedBins_[i]) result = true;
   }

   return result;
}

// ------------ method called once each job just before starting event loop  ------------
void 
CentralityFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
CentralityFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(CentralityFilter);
