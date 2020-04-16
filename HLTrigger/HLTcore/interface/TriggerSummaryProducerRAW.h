#ifndef HLTcore_TriggerSummaryProducerRAW_h
#define HLTcore_TriggerSummaryProducerRAW_h

/** \class TriggerSummaryProducerRAW
 *
 *  
 *  This class is an EDProducer making the HLT summary object for RAW
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"

#include <string>

namespace edm {
  class EventSetup;
}

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//
class TriggerSummaryProducerRAW : public edm::global::EDProducer<> {
public:
  explicit TriggerSummaryProducerRAW(const edm::ParameterSet&);
  ~TriggerSummaryProducerRAW() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  /// process name
  std::string pn_;

  edm::GetterOfProducts<trigger::TriggerFilterObjectWithRefs> getterOfProducts_;
  const edm::EDPutTokenT<trigger::TriggerEventWithRefs> putToken_;
};
#endif
