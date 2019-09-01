#ifndef MuonIsolation_CandViewExtractor_H
#define MuonIsolation_CandViewExtractor_H

#include <string>
#include <vector>

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"

namespace muonisolation {

  class CandViewExtractor : public reco::isodeposit::IsoDepositExtractor {
  public:
    CandViewExtractor(){};
    CandViewExtractor(const edm::ParameterSet& par, edm::ConsumesCollector&& iC);

    ~CandViewExtractor() override {}

    void fillVetos(const edm::Event& ev, const edm::EventSetup& evSetup, const reco::TrackCollection& cand) override {}

    /*  virtual reco::IsoDeposit::Vetos vetos(const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Candidate & cand) const;

  virtual reco::IsoDeposit::Vetos vetos(const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Track & cand) const;
*/

    void initEvent(const edm::Event& ev, const edm::EventSetup& evSetup) override;

    reco::IsoDeposit deposit(const edm::Event& ev,
                             const edm::EventSetup& evSetup,
                             const reco::Track& muon) const override {
      return depositFromObject(ev, evSetup, muon);
    }

    reco::IsoDeposit deposit(const edm::Event& ev,
                             const edm::EventSetup& evSetup,
                             const reco::Candidate& muon) const override {
      return depositFromObject(ev, evSetup, muon);
    }

  private:
    reco::IsoDeposit::Veto veto(const reco::IsoDeposit::Direction& dir) const;

    template <typename T>
    reco::IsoDeposit depositFromObject(const edm::Event& ev, const edm::EventSetup& evSetup, const T& cand) const;

    // Parameter set
    edm::EDGetTokenT<edm::View<reco::Candidate> > theCandViewToken;  // Track Collection Label
    std::string theDepositLabel;                                     // name for deposit
    edm::Handle<edm::View<reco::Candidate> > theCandViewH;           //cached handle
    edm::Event::CacheIdentifier_t theCacheID;                        //event cacheID
    double theDiff_r;                                                // transverse distance to vertex
    double theDiff_z;                                                // z distance to vertex
    double theDR_Max;                                                // Maximum cone angle for deposits
    double theDR_Veto;                                               // Veto cone angle
  };

}  // namespace muonisolation

#endif
