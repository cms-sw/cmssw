#ifndef MuonIsolationProducers_CandIsoDepositProducer_H
#define MuonIsolationProducers_CandIsoDepositProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include <DataFormats/RecoCandidate/interface/RecoCandidate.h>

#include "DataFormats/MuonReco/interface/MuIsoDepositFwd.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "RecoMuon/MuonIsolation/interface/MuIsoExtractor.h"

#include <string>

namespace edm { class Event; }
namespace edm { class EventSetup; }

class CandIsoDepositProducer : public edm::EDProducer {

public:
  CandIsoDepositProducer(const edm::ParameterSet&);

  virtual ~CandIsoDepositProducer();

  virtual void produce(edm::Event&, const edm::EventSetup&);
  
private:
  inline const reco::Track *extractTrack(const reco::Candidate &cand, reco::Track *dummyStorage) const; 
  enum TrackType { FakeT, BestT, StandAloneMuonT, CombinedMuonT, TrackT, GsfT, CandidateT };
  edm::ParameterSet theConfig;
  edm::InputTag theCandCollectionTag;
  TrackType     theTrackType;
  std::vector<std::string> theDepositNames;
  bool theMultipleDepositsFlag;
  muonisolation::MuIsoExtractor * theExtractor;

};
#endif
