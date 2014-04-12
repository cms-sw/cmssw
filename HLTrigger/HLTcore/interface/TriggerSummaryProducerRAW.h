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
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

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
class TriggerSummaryProducerRAW : public edm::EDProducer {
  
 public:
  explicit TriggerSummaryProducerRAW(const edm::ParameterSet&);
  ~TriggerSummaryProducerRAW();
  static  void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
  /// process name
  std::string pn_;

  edm::GetterOfProducts<trigger::TriggerFilterObjectWithRefs> getterOfProducts_;
};
#endif
