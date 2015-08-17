#include "GeneratorInterface/GenFilters/interface/LHEGenericFilter.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

LHEGenericFilter::LHEGenericFilter(const edm::ParameterSet& iConfig) :
  src_(iConfig.getParameter<edm::InputTag>("src")),
  numRequired_(iConfig.getParameter<int>("NumRequired")),
  acceptMore_(iConfig.getParameter<bool>("AcceptMore")),
  particleID_(iConfig.getParameter< std::vector<int> >("ParticleID")),
  totalEvents_(0), passedEvents_(0)
{
  //here do whatever other initialization is needed

}

LHEGenericFilter::~LHEGenericFilter()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


// ------------ method called to skim the data  ------------
bool LHEGenericFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle< LHEEventProduct > EvtHandle ;
  iEvent.getByLabel( src_ , EvtHandle ) ;
  
  totalEvents_++;
  int nFound = 0;
  
  for (int i = 0; i < EvtHandle->hepeup().NUP; ++i) {
     if (EvtHandle->hepeup().ISTUP[i] != 1) {
       continue;
     }    
    for (unsigned int j = 0; j < particleID_.size(); ++j) {
      if (particleID_[j] == 0 || abs(particleID_[j]) == abs(EvtHandle->hepeup().IDUP[i]) ) {
	nFound++;
	break; // only match a given particle once!
      }
    } // loop over targets
    
    std::cout << "test nFound = " << nFound << std::endl;
    if (acceptMore_ && nFound == numRequired_) break; // stop looking if we don't mind having more
  } // loop over particles
  
  if (nFound == numRequired_) {
    passedEvents_++;
    return true;
  } else {
    return false;
  }
  
}

// ------------ method called once each job just after ending the event loop  ------------
void LHEGenericFilter::endJob() {
  edm::LogInfo("LHEGenericFilter") << "=== Results of LHEGenericFilter: passed "
                                        << passedEvents_ << "/" << totalEvents_ << " events" << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(LHEGenericFilter);

