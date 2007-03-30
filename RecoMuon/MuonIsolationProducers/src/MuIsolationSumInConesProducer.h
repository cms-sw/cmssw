#ifndef MuonIsolationProducers_MuIsolationSumInConesProducer_H
#define MuonIsolationProducers_MuIsolationSumInConesProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/MuonIsolation/interface/MuIsoExtractor.h"
#include <string>

namespace edm { class Event; }
namespace edm { class EventSetup; }

class MuIsolationSumInConesProducer : public edm::EDProducer {

public:

  MuIsolationSumInConesProducer(const edm::ParameterSet&);

  virtual ~MuIsolationSumInConesProducer();

  virtual void produce(edm::Event&, const edm::EventSetup&);
  
private:
  struct DepositConf { edm::InputTag tag; double weight; double threshold;};
  struct VetoCuts { 
    bool selectAll; 
    double muAbsEtaMax; 
    double muPtMin;
    double muAbsZMax;
    double muD0Max;
  };

  edm::ParameterSet theConfig;
  std::vector<DepositConf> theDepositConfs;
  std::vector<std::pair<double, std::string> > theConeSizes;
  bool theProduceFloat;
  bool theProduceInt;
  

  ///choose which muon vetos should be removed from all deposits  
  bool theRemoveOtherVetos;
  VetoCuts theVetoCuts;
};
#endif
