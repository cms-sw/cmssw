#ifndef HLTPrescaler_H
#define HLTPrescaler_H

/** \class HLTPrescaler
 *
 *  
 *  This class is an EDFilter implementing an HLT
 *  Prescaler module with associated book keeping.
 *
 *
 *  \author Martin Grunewald
 *  \author Philipp Schieferdecker
 */

#include <atomic>
#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PrescaleService/interface/PrescaleService.h"

// legacy/stage-1 L1T:
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

// stage-2 L1T:
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"

namespace edm {
  class ConfigurationDescriptions;
}

namespace trigger {
  struct Efficiency {
    Efficiency() : eventCount_(0), acceptCount_(0) {}
    mutable std::atomic<unsigned int> eventCount_;
    mutable std::atomic<unsigned int> acceptCount_;
  };
}  // namespace trigger

class HLTPrescaler : public edm::stream::EDFilter<edm::GlobalCache<trigger::Efficiency> > {
public:
  //
  // construction/destruction
  //
  explicit HLTPrescaler(edm::ParameterSet const& iConfig, const trigger::Efficiency* efficiency);
  ~HLTPrescaler() override;

  static std::unique_ptr<trigger::Efficiency> initializeGlobalCache(edm::ParameterSet const&) {
    return std::make_unique<trigger::Efficiency>();
  }

  //
  // member functions
  //
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void beginLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& iSetup) override;
  bool filter(edm::Event& iEvent, edm::EventSetup const& iSetup) override;
  void endStream() override;
  static void globalEndJob(const trigger::Efficiency* efficiency);

private:
  //
  //member data
  //

  /// l1 prescale set index
  unsigned int prescaleSet_;

  /// accept one in prescaleFactor_; 0 means never to accept an event
  unsigned int prescaleFactor_;

  /// event counter
  unsigned int eventCount_;

  /// accept counter
  unsigned int acceptCount_;

  /// initial offset
  unsigned int offsetCount_;
  unsigned int offsetPhase_;

  /// prescale service
  edm::service::PrescaleService* prescaleService_;

  /// check for (re)initialization of the prescale
  bool newLumi_;

  /// GT payload, to extract the prescale column index
  edm::InputTag gtDigiTag_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> gtDigi1Token_;
  edm::EDGetTokenT<GlobalAlgBlkBxCollection> gtDigi2Token_;

  /// "seed" used to initialize the prescale counter
  static const unsigned int prescaleSeed_;
};

#endif
