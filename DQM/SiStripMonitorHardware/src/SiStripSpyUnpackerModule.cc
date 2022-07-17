/*\ \file SiStripSpyUnpackerModule.cc
 * \brief Source code for the spy unpacking plugin module.
 */

// CMSSW includes
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// Needed for the FED cabling
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

// Needed for the FED raw data collection
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

// For the digi stuff.
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

// For the unpacking object.
#include "DQM/SiStripMonitorHardware/interface/SiStripSpyUnpacker.h"
#include "DQM/SiStripMonitorHardware/interface/SiStripSpyUtilities.h"

// Standard includes
#include <memory>
#include <utility>
#include <string>
#include <cstdint>

namespace sistrip {
  class SpyUnpacker;
}  // namespace sistrip

using edm::LogError;
using edm::LogInfo;

namespace sistrip {

  /*!
     * @file DQM/SiStripMonitorHardware/interface/SiStripSpyUnpackerModule.cc
     * @class SiStripSpyUnpackerModule 
     *
     * @brief A plug-in module that takes a FEDRawDataCollection as input
     * from the Event and creates EDProducts containing StripDigis from spy channel 
     * data.
     */
  class SpyUnpackerModule : public edm::global::EDProducer<> {
  public:
    SpyUnpackerModule(const edm::ParameterSet&);
    ~SpyUnpackerModule() override;
    void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  private:
    static const char* const msgLb_;

    // Data members
    //--------------
    // Configuration
    std::vector<uint32_t> fed_ids_;     //!< Vector of FED IDs to examine (FEDs).
    const edm::InputTag productLabel_;  //!< The product label of the FEDRawDataCollection input.
    edm::EDGetTokenT<FEDRawDataCollection> productToken_;
    const bool allowIncompleteEvents_;  //!< Allow inconsistent (by event count, APV address) event storage.
    const bool storeCounters_;          //!< True = store L1ID and TotalEventCount by FED key.
    const bool storeScopeRawDigis_;     //!< True = store the scope mode raw digis.
                                        // Unpacking
    const SpyUnpacker unpacker_;        //!<

    //utilities for cabling etc...
    const edm::ESGetToken<SiStripFedCabling, SiStripFedCablingRcd> fedCablingToken_;
  };  // end of SpyUnpackerModule class.

}  // end of namespace sistrip.

namespace sistrip {

  const char* const SpyUnpackerModule::msgLb_ = "SiStripSpyUnpackerModule";

  SpyUnpackerModule::SpyUnpackerModule(const edm::ParameterSet& pset)
      : fed_ids_(pset.getParameter<std::vector<uint32_t> >("FEDIDs")),
        productLabel_(pset.getParameter<edm::InputTag>("InputProductLabel")),
        allowIncompleteEvents_(pset.getParameter<bool>("AllowIncompleteEvents")),
        storeCounters_(pset.getParameter<bool>("StoreCounters")),
        storeScopeRawDigis_(pset.getParameter<bool>("StoreScopeRawDigis")),
        unpacker_(allowIncompleteEvents_),
        fedCablingToken_(esConsumes<>()) {
    productToken_ = consumes<FEDRawDataCollection>(productLabel_);

    if ((fed_ids_.empty())) {
      LogInfo(msgLb_) << "No FED IDs specified, so will try to unpack all FEDs with data" << std::endl;
      fed_ids_.reserve(FEDNumbering::MAXSiStripFEDID - FEDNumbering::MINSiStripFEDID + 1);
      for (uint32_t ifed = FEDNumbering::MINSiStripFEDID; ifed <= FEDNumbering::MAXSiStripFEDID; ifed++) {
        fed_ids_.push_back(ifed);
      }
    }  // end of FED ID specified check.

    if (edm::isDebugEnabled())
      LogTrace(msgLb_) << "[" << __func__ << "]:"
                       << " Constructing object...";

    if (storeScopeRawDigis_)
      produces<edm::DetSetVector<SiStripRawDigi> >("ScopeRawDigis");

    if (storeCounters_) {
      produces<std::vector<uint32_t> >("L1ACount");
      produces<std::vector<uint32_t> >("TotalEventCount");
    }

    produces<uint32_t>("GlobalRunNumber");

  }  // end of SpyUnpackerModule constructor.

  SpyUnpackerModule::~SpyUnpackerModule() {
    if (edm::isDebugEnabled()) {
      LogTrace("SiStripSpyUnpacker") << "[sistrip::SpyUnpackerModule::" << __func__ << "]"
                                     << " Destructing object...";
    }
  }

  /*! \brief Scope mode digis and event counter producer.
   *  Retrieves cabling map from EventSetup and FEDRawDataCollection
   *  from Event, creates a DetSetVector of SiStripRawDigis, uses the
   *  SiStripSpyUnpacker class to fill the DetSetVector, and
   *  attaches the container to the Event.
   */
  void SpyUnpackerModule::produce(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
    const SiStripFedCabling* fedCabling = &setup.getData(fedCablingToken_);
    //retrieve FED raw data (by label, which is "source" by default)
    edm::Handle<FEDRawDataCollection> buffers;
    event.getByToken(productToken_, buffers);

    //create container for digis
    std::unique_ptr<edm::DetSetVector<SiStripRawDigi> > digis(new edm::DetSetVector<SiStripRawDigi>);

    //if necessary, create container for event counters
    std::unique_ptr<std::vector<uint32_t> > pTotalCounts(new std::vector<uint32_t>);
    std::unique_ptr<std::vector<uint32_t> > pL1ACounts(new std::vector<uint32_t>);
    //and for run number
    std::unique_ptr<uint32_t> pGlobalRun(new uint32_t);
    //create digis
    // Using FED IDs...
    unpacker_.createDigis(
        *fedCabling, *buffers, digis.get(), fed_ids_, pTotalCounts.get(), pL1ACounts.get(), pGlobalRun.get());

    // Add digis to event
    if (storeScopeRawDigis_)
      event.put(std::move(digis), "ScopeRawDigis");

    //add counters to event
    if (storeCounters_) {
      event.put(std::move(pTotalCounts), "TotalEventCount");
      event.put(std::move(pL1ACounts), "L1ACount");
    }

    //add global run to the event
    event.put(std::move(pGlobalRun), "GlobalRunNumber");

  }  // end of SpyUnpackerModule::produce method.

}  // namespace sistrip

typedef sistrip::SpyUnpackerModule SiStripSpyUnpackerModule;
DEFINE_FWK_MODULE(SiStripSpyUnpackerModule);
