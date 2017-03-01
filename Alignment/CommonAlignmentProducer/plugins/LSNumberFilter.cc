//#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"



//
// class declaration
//



class LSNumberFilter : public edm::EDFilter {
public:
  explicit LSNumberFilter(const edm::ParameterSet&);
  ~LSNumberFilter();

private:

  virtual void beginJob() override ;
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override ;

  unsigned int minLS;
};



LSNumberFilter::LSNumberFilter(const edm::ParameterSet& iConfig):
  minLS(iConfig.getUntrackedParameter<unsigned>("minLS",21))
{}


LSNumberFilter::~LSNumberFilter()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool LSNumberFilter::filter(edm::Event& iEvent,
			    const edm::EventSetup& iSetup) {

  if(iEvent.luminosityBlock() < minLS) return false;
  
  return true;

}

// ------------ method called once each job just before starting event loop  ------------
void
LSNumberFilter::beginJob()
{}

// ------------ method called once each job just after ending the event loop  ------------
void
LSNumberFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(LSNumberFilter);
