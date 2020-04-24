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
}

HLTTauCertifier::~HLTTauCertifier() = default;

//--------------------------------------------------------
void HLTTauCertifier::dqmEndJob(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter) {
  int warnings=0;
  int errors=0;
  double response=1.0;
  
  for(auto const & inputME : inputMEs_)
    {
      MonitorElement *monElement = iGetter.get(inputME);
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
  iBooker.setCurrentFolder(targetFolder_);
  MonitorElement *certME = iBooker.bookFloat(targetME_);
  certME->Fill(response);
}
