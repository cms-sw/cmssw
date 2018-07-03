#ifndef RecoTBCalo_EcalTBHodoscopeReconstructor_EcalTBHodoscopeRecInfoProducer_HH
#define RecoTBCalo_EcalTBHodoscopeReconstructor_EcalTBHodoscopeRecInfoProducer_HH

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTBCalo/EcalTBHodoscopeReconstructor/interface/EcalTBHodoscopeRecInfoAlgo.h"

#include <vector>

class EcalTBHodoscopeRecInfoProducer : public edm::EDProducer {

 public:

  explicit EcalTBHodoscopeRecInfoProducer(const edm::ParameterSet& ps);
  ~EcalTBHodoscopeRecInfoProducer() override ;
  void produce(edm::Event& e, const edm::EventSetup& es) override;

 private:

  std::string rawInfoProducer_; // name of module/plugin/producer making digis
  std::string rawInfoCollection_; // secondary name given to collection of digis
  std::string recInfoCollection_; // secondary name to be given to collection of hits

  int fitMethod_;
  EcalTBHodoscopeRecInfoAlgo *algo_;

};
#endif
