#ifndef HLTcore_TriggerSummaryProducerAOD_h
#define HLTcore_TriggerSummaryProducerAOD_h

/** \class TriggerSummaryProducerAOD
 *
 *  
 *  This class is an EDProducer making the HLT summary object for AOD
 *
 *  $Date: 2007/08/07 18:42:18 $
 *  $Revision: 1.13 $
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
class TriggerSummaryProducerAOD : public edm::EDProducer {
  
 public:
  explicit TriggerSummaryProducerAOD(const edm::ParameterSet&);
  ~TriggerSummaryProducerAOD();
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  /// process name
  std::string pn_;
  /// selector for getMany methods
  edm::ProcessNameSelector selector_;
  /// the pointer to the current TriggerNamesService
  edm::service::TriggerNamesService* tns_;
  /// list of L3 collection labels
  std::vector<edm::InputTag> collections_;
  /// list of L3 filter labels
  std::vector<edm::InputTag> filters_;
  /// global map for indices of L3 collections: offset per collection
  std::map<edm::ProductID,int> offset_;
  /// handles to the filter objects
  std::vector<edm::Handle<trigger::TriggerFilterObjectWithRefs> > fobs_;

};
#endif
