#ifndef HLTcore_TriggerSummaryProducerRAW_h
#define HLTcore_TriggerSummaryProducerRAW_h

/** \class TriggerSummaryProducerRAW
 *
 *  
 *  This class is an EDProducer making the HLT summary object for RAW
 *
 *  $Date: 2012/08/09 20:00:18 $
 *  $Revision: 1.2 $
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

//
// class declaration
//
class TriggerSummaryProducerRAW : public edm::EDProducer {
  
 public:
  explicit TriggerSummaryProducerRAW(const edm::ParameterSet&);
  ~TriggerSummaryProducerRAW();
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
  /// process name
  std::string pn_;

  edm::GetterOfProducts<trigger::TriggerFilterObjectWithRefs> getterOfProducts_;
};
#endif
