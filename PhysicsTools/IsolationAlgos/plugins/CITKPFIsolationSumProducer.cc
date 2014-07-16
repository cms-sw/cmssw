#ifndef PhysicsTools_IsolationAlgos_CITKIsolationSumProducer_H
#define PhysicsTools_IsolationAlgos_CITKIsolationSumProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "PhysicsTools/IsolationAlgos/interface/EventDependentAbsVeto.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/IsolationAlgos/interface/CITKIsolationConeDefinitionBase.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include <string>
#include <unordered_map>

namespace edm { class Event; }
namespace edm { class EventSetup; }

class CITKPFIsolationSumProducer : public edm::EDProducer {

public:  
  CITKPFIsolationSumProducer(const edm::ParameterSet&);

  virtual ~CITKPFIsolationSumProducer();

  void produce(edm::Event&, const edm::EventSetup&) override final;

private:  
  // datamembers
  std::vector<SingleDeposit> sources_;

};

DEFINE_FWK_MODULE(  );

#endif
