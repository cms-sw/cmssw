#ifndef MuonIsolationProducers_CandIsoDepositProducer_H
#define MuonIsolationProducers_CandIsoDepositProducer_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include <DataFormats/RecoCandidate/interface/RecoCandidate.h>

#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"

#include <string>

namespace edm { class Event; }
namespace edm { class EventSetup; }

class CandIsoDepositProducer : public edm::stream::EDProducer<> {

public:
  CandIsoDepositProducer(const edm::ParameterSet&);

  virtual ~CandIsoDepositProducer();

  virtual void produce(edm::Event&, const edm::EventSetup&);

private:
  inline const reco::Track *extractTrack(const reco::Candidate &cand, reco::Track *dummyStorage) const;
  enum TrackType { FakeT, BestT, StandAloneMuonT, CombinedMuonT, TrackT, GsfT, CandidateT };
  edm::ParameterSet theConfig;
  edm::EDGetTokenT< edm::View<reco::Candidate> > theCandCollectionToken;
  TrackType     theTrackType;
  std::vector<std::string> theDepositNames;
  bool theMultipleDepositsFlag;
  reco::isodeposit::IsoDepositExtractor * theExtractor;

};
#endif
