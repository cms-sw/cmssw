#include "DQMOffline/Trigger/interface/HLTTauCertifier.h"

using namespace std;
using namespace edm;

//
// constructors and destructor
//
HLTTauCertifier::HLTTauCertifier( const edm::ParameterSet& ps )
{
  targetFolder_       = ps.getParameter<string>("targetDir");
  targetME_           = ps.getParameter<string>("targetME");
  inputMEs_           = ps.getParameter<vector<string> >("inputMEs");
  setBadRunOnWarnings_ = ps.getParameter<bool>("setBadRunOnWarnings");
  setBadRunOnErrors_   = ps.getParameter<bool>("setBadRunOnErrors");

   dbe_ = &*edm::Service<DQMStore>();

}

HLTTauCertifier::~HLTTauCertifier()
{
   
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}


//--------------------------------------------------------
void 
HLTTauCertifier::beginJob(){

}

//--------------------------------------------------------
void HLTTauCertifier::beginRun(const edm::Run& r, const EventSetup& context) {

}

//--------------------------------------------------------
void HLTTauCertifier::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
				      const EventSetup& context) {
  
}

// ----------------------------------------------------------
void 
HLTTauCertifier::analyze(const Event& iEvent, const EventSetup& iSetup )
{  

}




//--------------------------------------------------------
void HLTTauCertifier::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
				    const EventSetup& context) {
}
//--------------------------------------------------------
void HLTTauCertifier::endRun(const Run& r, const EventSetup& context){
  if(dbe_) {
    int warnings=0;
    int errors=0;
    double response=1.0;
    
    for(unsigned int i=0;i<inputMEs_.size();++i)
      {
	MonitorElement *monElement = dbe_->get(inputMEs_.at(i));
	if(monElement)
	  {
	    warnings+=monElement->getQWarnings().size();
	    errors+=monElement->getQErrors().size();
	  }
      }
    if(setBadRunOnWarnings_ && warnings>0) 
      response=0.0;

    if(setBadRunOnErrors_ && errors>0) 
      response=0.0;

    //OK SAVE THE FINAL RESULT	
    dbe_->setCurrentFolder(targetFolder_);
    MonitorElement *certME = dbe_->bookFloat(targetME_);
    certME->Fill(response);
  }

}
//--------------------------------------------------------
void HLTTauCertifier::endJob(){
  return;
}



