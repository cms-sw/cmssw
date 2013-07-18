#ifndef PhysicsTools_IsolationAlgos_PFTauExtractor_H
#define PhysicsTools_IsolationAlgos_PFTauExtractor_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"

class PFTauExtractor : public reco::isodeposit::IsoDepositExtractor 
{
 public:

  explicit PFTauExtractor(const edm::ParameterSet&);
  virtual ~PFTauExtractor(){}

  /// definition of pure virtual functions inherited from IsoDepositExtractor base-class
  virtual void fillVetos(const edm::Event&, const edm::EventSetup&, const reco::TrackCollection&) { }
  virtual reco::IsoDeposit deposit(const edm::Event& evt, const edm::EventSetup& es, const reco::Track& track) const { 
    return depositFromObject(evt, es, track);
  }
  virtual reco::IsoDeposit deposit(const edm::Event& evt, const edm::EventSetup& es, const reco::Candidate& candidate) const { 
    return depositFromObject(evt, es, candidate);
  }

 private:

  /// configuration parameters
  edm::InputTag tauSource_;
  edm::InputTag candidateSource_;
  double maxDxyTrack_;
  double maxDzTrack_;
  double dRmatchPFTau_;
  double dRVetoCone_;
  double dRIsoCone_;
  double dRvetoPFTauSignalConeConstituents_;

  /// private member function for computing the IsoDeposits 
  /// in case of reco::Track as well as in case of reco::Canididate input
  template<typename T>
  reco::IsoDeposit depositFromObject(const edm::Event&, const edm::EventSetup&, const T&) const;

};

#endif
