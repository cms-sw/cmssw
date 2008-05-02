///////////////////////////////////////////////////////////////////////////////
//
// HLTPrescaler
// ------------
//
//           04/25/2008 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
///////////////////////////////////////////////////////////////////////////////


#include "HLTrigger/HLTcore/interface/HLTPrescaler.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

///////////////////////////////////////////////////////////////////////////////
// construction/destruction
///////////////////////////////////////////////////////////////////////////////

//_____________________________________________________________________________
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

//_____________________________________________________________________________    
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
    const unsigned int oldPrescale(prescaleFactor_);
    prescaleFactor_ = prescaleService_->getPrescale(*pathName());
    if (prescaleFactor_!=oldPrescale)
      edm::LogInfo("ChangedPrescale")
	<< "lumiBlockNb="<<lb.id().luminosityBlock() << ", "
	<< "path="<<*pathName()<<": "
	<< prescaleFactor_ << " [" <<oldPrescale<<"]";
  }
  return true;
}


//_____________________________________________________________________________
bool HLTPrescaler::filter(edm::Event&, const edm::EventSetup&)
{
  const bool result ( (prescaleFactor_==0) ? 
		      false : (eventCount_%prescaleFactor_==0) );
  ++eventCount_;
  if (result) ++acceptCount_;
  return result;
}


//_____________________________________________________________________________
void HLTPrescaler::endJob()
{
  edm::LogInfo("PrescaleSummary")
    << acceptCount_<< "/" <<eventCount_
    << " ("
    << 100.*acceptCount_/static_cast<double>(std::max(1u,eventCount_))
    << "% of events accepted).";
  return;
}
