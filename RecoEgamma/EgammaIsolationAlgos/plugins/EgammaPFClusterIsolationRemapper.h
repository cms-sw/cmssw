#ifndef EgammaIsolationRemappers_EgammaPFClusterIsolationRemapper_h
#define EgammaIsolationRemappers_EgammaPFClusterIsolationRemapper_h

//*****************************************************************************
// File:      EgammaPFClusterIsolationRemapper.h
// ----------------------------------------------------------------------------
// OrigAuth:  Yutaro Iiyama
// Institute: MIT
//*****************************************************************************

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Ref.h"

template<typename C>
class EgammaPFClusterIsolationRemapper : public edm::stream::EDProducer<> {
 public:
  EgammaPFClusterIsolationRemapper(const edm::ParameterSet&);
  ~EgammaPFClusterIsolationRemapper();
 
  void produce(edm::Event&, const edm::EventSetup&);
 private:
  typedef typename edm::refhelper::ValueTrait<C>::value Value;
  typedef edm::Ref<C, Value, typename edm::refhelper::FindTrait<C, Value>::value> CRef;
  typedef edm::ValueMap<CRef> CRefMap;
  typedef edm::ValueMap<float> FloatMap;

  edm::EDGetTokenT<C> emObjectProducer_;
  edm::EDGetTokenT<CRefMap> newToOldObjectMap_;
  edm::EDGetTokenT<FloatMap> isolationMap_;
};

#endif
