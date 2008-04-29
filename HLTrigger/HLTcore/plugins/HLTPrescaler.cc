////////////////////////////////////////////////////////////////////////////////
//
// HLTPrescaler
// ------------
//
//            04/25/2008 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "HLTrigger/HLTcore/interface/HLTPrescaler.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
HLTPrescaler::HLTPrescaler(edm::ParameterSet const& iConfig)
  : prescaleFactor_(1)
  , eventCount_(0)
  , acceptCount_(0)
  , prescaleService_(0)
{
  if(edm::Service<edm::service::PrescaleService>().isAvailable())
    prescaleService_ = edm::Service<edm::service::PrescaleService>().operator->();
  else 
    LogDebug("NoPrescaleService")<<"PrescaleService unavailable, prescaleFactor=1!";
}

//______________________________________________________________________________    
HLTPrescaler::~HLTPrescaler()
{
  
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
bool HLTPrescaler::beginLuminosityBlock(edm::LuminosityBlock & lb,
					edm::EventSetup const& iSetup)
{
  if (prescaleService_) {
    unsigned int oldPrescale = prescaleFactor_;
    prescaleFactor_ = prescaleService_->getPrescale(*pathName());
    if (prescaleFactor_!=oldPrescale)
      edm::LogInfo("ChangedPrescale")
	<<"lumiBlockNb="<<lb.id().luminosityBlock()<<", "
	<<"path="<<*pathName()<<": "<<prescaleFactor_<<" ["<<oldPrescale<<"]";
  }
  return true;
}


//______________________________________________________________________________
bool HLTPrescaler::filter(edm::Event&, const edm::EventSetup&)
{
  ++eventCount_;
  bool result = (prescaleFactor_==0) ? false : (eventCount_%prescaleFactor_==0);
  if (result) acceptCount_++;
  return result;
}


//______________________________________________________________________________
void HLTPrescaler::endJob()
{
  edm::LogInfo("PrescaleSummary")
    <<acceptCount_<<"/"<<eventCount_
    <<" ("<<100.*acceptCount_/(double)eventCount_<<"%) events accepted";
}
