#ifndef HLTcore_TriggerSummaryProducerAOD_h
#define HLTcore_TriggerSummaryProducerAOD_h

/** \class TriggerSummaryProducerAOD
 *
 *  
 *  This class is an EDProducer making the HLT summary object for AOD
 *
 *  $Date: 2007/12/06 20:37:03 $
 *  $Revision: 1.2 $
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
#include "DataFormats/HLTReco/interface/TriggerObject.h"
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


  // additional

  template <typename C>
  void fillTriggerObjects(const edm::Event& );

  template <typename C>
  void fillFilterKeys(const std::vector<edm::Ref<C> >&);

  template <typename C>
  trigger::size_type fillMask(const std::vector<edm::Handle<C> >& ,  const std::vector<edm::InputTag>& );

 private:
  /// process name
  std::string pn_;
  /// selector for getMany methods
  edm::ProcessNameSelector selector_;
  /// the pointer to the current TriggerNamesService
  edm::service::TriggerNamesService* tns_;
  /// list of L3 collection labels
  std::vector<edm::InputTag> collectionTags_;
  /// list of L3 filter labels
  std::vector<edm::InputTag> filterTags_;

  /// trigger object collection
  trigger::TriggerObjectCollection toc_;
  /// global map for indices into toc_: offset per input L3 collection
  std::map<edm::ProductID,int> offset_;

  /// handles to the filter objects
  std::vector<edm::Handle<trigger::TriggerFilterObjectWithRefs> > fobs_;
  /// keys
  trigger::Keys keys_;

  /// packing decision
  std::vector<bool> mask_;

};
#endif
