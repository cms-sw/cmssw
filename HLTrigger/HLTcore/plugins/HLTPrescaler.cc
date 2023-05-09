///////////////////////////////////////////////////////////////////////////////
//
// HLTPrescaler
// ------------
//
//           04/25/2008 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/PlaceInPathContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "HLTrigger/HLTcore/interface/HLTPrescaler.h"

///////////////////////////////////////////////////////////////////////////////
// initialize static member variables
///////////////////////////////////////////////////////////////////////////////

const unsigned int HLTPrescaler::prescaleSeed_ = 65537;

///////////////////////////////////////////////////////////////////////////////
// construction/destruction
///////////////////////////////////////////////////////////////////////////////

//_____________________________________________________________________________
HLTPrescaler::HLTPrescaler(edm::ParameterSet const& iConfig, const trigger::Efficiency* efficiency)
    : prescaleSet_(0),
      prescaleFactor_(1),
      eventCount_(0),
      acceptCount_(0),
      offsetCount_(0),
      offsetPhase_(iConfig.getParameter<unsigned int>("offset")),
      prescaleService_(nullptr),
      newLumi_(true),
      gtDigiTag_(iConfig.getParameter<edm::InputTag>("L1GtReadoutRecordTag")),
      gtDigi1Token_(consumes<L1GlobalTriggerReadoutRecord>(gtDigiTag_)),
      gtDigi2Token_(consumes<GlobalAlgBlkBxCollection>(gtDigiTag_)) {
  if (edm::Service<edm::service::PrescaleService>().isAvailable())
    prescaleService_ = edm::Service<edm::service::PrescaleService>().operator->();
  else
    LogDebug("NoPrescaleService") << "PrescaleService unavailable, prescaleFactor=1!";
}

//_____________________________________________________________________________
HLTPrescaler::~HLTPrescaler() = default;

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

void HLTPrescaler::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<unsigned int>("offset", 0);
  desc.add<edm::InputTag>("L1GtReadoutRecordTag", edm::InputTag("hltGtStage2Digis"));
  descriptions.add("hltPrescaler", desc);
}

//______________________________________________________________________________
void HLTPrescaler::beginLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& iSetup) {
  newLumi_ = true;
}

//_____________________________________________________________________________
bool HLTPrescaler::filter(edm::Event& iEvent, const edm::EventSetup&) {
  // during the first event of a LumiSection, read from the GT the prescale index for this
  // LumiSection and get the corresponding prescale factor from the PrescaleService
  if (newLumi_) {
    newLumi_ = false;

    bool needsInit(eventCount_ == 0);

    if (prescaleService_) {
      std::string const& pathName = iEvent.moduleCallingContext()->placeInPathContext()->pathContext()->pathName();
      const unsigned int oldSet(prescaleSet_);
      const unsigned int oldPrescale(prescaleFactor_);

      edm::Handle<GlobalAlgBlkBxCollection> handle2;
      iEvent.getByToken(gtDigi2Token_, handle2);
      if (handle2.isValid()) {
        if (not handle2->isEmpty(0)) {
          prescaleSet_ = static_cast<unsigned int>(handle2->begin(0)->getPreScColumn());
          prescaleFactor_ = prescaleService_->getPrescale(prescaleSet_, pathName);
        } else {
          edm::LogWarning("HLT")
              << "Cannot read prescale column index from GT2 data: using default as defined by configuration or DAQ";
          prescaleFactor_ = prescaleService_->getPrescale(pathName);
        }
      } else {
        edm::Handle<L1GlobalTriggerReadoutRecord> handle1;
        iEvent.getByToken(gtDigi1Token_, handle1);
        if (handle1.isValid()) {
          prescaleSet_ = handle1->gtFdlWord().gtPrescaleFactorIndexAlgo();
          // gtPrescaleFactorIndexTech() is also available
          // by construction, they should always return the same index
          prescaleFactor_ = prescaleService_->getPrescale(prescaleSet_, pathName);
        } else {
          edm::LogWarning("HLT")
              << "Cannot read prescale column index from GT1 data: using default as defined by configuration or DAQ";
          prescaleFactor_ = prescaleService_->getPrescale(pathName);
        }
      }

      if (prescaleSet_ != oldSet) {
        edm::LogInfo("ChangedPrescale") << "lumiBlockNb = " << iEvent.getLuminosityBlock().id().luminosityBlock()
                                        << ", set = " << prescaleSet_ << " [" << oldSet << "]"
                                        << ", path = " << pathName << ": " << prescaleFactor_ << " [" << oldPrescale
                                        << "]";
        // reset the prescale counter
        needsInit = true;
      }
    }

    if (needsInit && (prescaleFactor_ != 0)) {
      // initialize the prescale counter to the first event number multiplied by a big "seed"
      offsetCount_ = ((uint64_t)(iEvent.id().event() + offsetPhase_) * prescaleSeed_) % prescaleFactor_;
    }
  }

  const bool result((prescaleFactor_ == 0) ? false : ((eventCount_ + offsetCount_) % prescaleFactor_ == 0));

  ++eventCount_;
  if (result)
    ++acceptCount_;
  return result;
}

//_____________________________________________________________________________
void HLTPrescaler::endStream() {
  //since these are std::atomic, it is safe to increment them
  // even if multiple endStreams are being called.
  globalCache()->eventCount_ += eventCount_;
  globalCache()->acceptCount_ += acceptCount_;
  return;
}

//_____________________________________________________________________________
void HLTPrescaler::globalEndJob(const trigger::Efficiency* efficiency) {
  unsigned int accept(efficiency->acceptCount_);
  unsigned int event(efficiency->eventCount_);
  edm::LogInfo("PrescaleSummary") << accept << "/" << event << " ("
                                  << 100. * accept / static_cast<double>(std::max(1u, event))
                                  << "% of events accepted).";
  return;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTPrescaler);
