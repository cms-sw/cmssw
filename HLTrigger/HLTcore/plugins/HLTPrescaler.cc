///////////////////////////////////////////////////////////////////////////////
//
// HLTPrescaler
// ------------
//
//           04/25/2008 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
///////////////////////////////////////////////////////////////////////////////


#include "HLTrigger/HLTcore/interface/HLTPrescaler.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

///////////////////////////////////////////////////////////////////////////////
// initialize static member variables
///////////////////////////////////////////////////////////////////////////////

const unsigned int HLTPrescaler::prescaleSeed_ = 65537;

///////////////////////////////////////////////////////////////////////////////
// construction/destruction
///////////////////////////////////////////////////////////////////////////////

//_____________________________________________________________________________
HLTPrescaler::HLTPrescaler(edm::ParameterSet const& iConfig)
  : prescaleFactor_(1)
  , eventCount_(0)
  , acceptCount_(0)
  , offsetCount_(0)
  , offsetPhase_(iConfig.existsAs<unsigned int>("offset") ? iConfig.getParameter<unsigned int>("offset") : 0)
  , prescaleService_(0)
  , newLumi_(true)
  , gtDigi_ (iConfig.getParameter<edm::InputTag>("L1GtReadoutRecordTag"))
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
  newLumi_ = true;

  return true;
}


//_____________________________________________________________________________
bool HLTPrescaler::filter(edm::Event& iEvent, const edm::EventSetup&)
{
  // during the first event of a LumiSection, read from the GT the prescale index for this
  // LumiSection and get the corresponding prescale factor from the PrescaleService
  if (newLumi_) {
    newLumi_ = false;

    bool needsInit (eventCount_==0);

    if (prescaleService_) {
      const unsigned int oldPrescale(prescaleFactor_);

      edm::Handle<L1GlobalTriggerReadoutRecord> handle;
      iEvent.getByLabel(gtDigi_ , handle);
      if (handle.isValid()) {
        unsigned int index = handle->gtFdlWord().gtPrescaleFactorIndexAlgo();
        // gtPrescaleFactorIndexTech() is also available
        // by construction, they should always return the same index
        prescaleFactor_ = prescaleService_->getPrescale(index, *pathName());
      } else {
        edm::LogWarning("HLT") << "Cannot read prescale column index from GT data: using default as defined by configuration or DAQ";
        prescaleFactor_ = prescaleService_->getPrescale(*pathName());
      }

      if (prescaleFactor_ != oldPrescale) {
        edm::LogInfo("ChangedPrescale")
          << "lumiBlockNb="<< iEvent.getLuminosityBlock().id().luminosityBlock() << ", "
          << "path="<<*pathName()<<": "
          << prescaleFactor_ << " [" <<oldPrescale<<"]";
        // reset the prescale counter
        needsInit = true;
      }
    }

    if (needsInit && (prescaleFactor_ != 0)) {
      // initialize the prescale counter to the first event number multiplied by a big "seed"
      offsetCount_ = ((uint64_t) (iEvent.id().event() + offsetPhase_) * prescaleSeed_) % prescaleFactor_;
    }
  }

  const bool result ( (prescaleFactor_ == 0) ? 
		      false : ((eventCount_ + offsetCount_) % prescaleFactor_ == 0) );

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
