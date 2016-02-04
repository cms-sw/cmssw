#ifndef HLTcore_TriggerSummaryProducerRAW_h
#define HLTcore_TriggerSummaryProducerRAW_h

/** \class TriggerSummaryProducerRAW
 *
 *  
 *  This class is an EDProducer making the HLT summary object for RAW
 *
 *  $Date: 2007/12/06 08:27:31 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include<string>
#include<vector>

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
  /// selector for getMany methods
  edm::ProcessNameSelector selector_;
  /// the pointer to the current TriggerNamesService
  edm::service::TriggerNamesService* tns_;
  /// handles to the filter objects
  std::vector<edm::Handle<trigger::TriggerFilterObjectWithRefs> > fobs_;

};
#endif
