#ifndef DataFormats_L1TParticleFlow_HPSPFTau_H
#define DataFormats_L1TParticleFlow_HPSPFTau_H

#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"  // l1t::PFCandidate, l1t::PFCandidateRef, l1t::PFCandidateRefVector
#include "DataFormats/L1TParticleFlow/interface/PFJet.h"    // l1t::PFJet, l1t::PFJetCollection, l1t::PFJetRef
#include "DataFormats/Candidate/interface/LeafCandidate.h"  // reco::LeafCandidate
#include "DataFormats/Candidate/interface/Particle.h"       // reco::Particle::LorentzVector
#include "DataFormats/L1TCorrelator/interface/TkPrimaryVertex.h"

#include <ostream>

namespace l1t {

  class HPSPFTau : public reco::LeafCandidate {
  public:
    /// default constructor
    HPSPFTau();

    /// destructor
    ~HPSPFTau() override;

    /// accessor functions for reco level quantities
    bool isChargedPFCandSeeded() const { return seedChargedPFCand_.isNonnull(); }
    bool isPFJetSeeded() const { return seedPFJet_.isNonnull(); }

    const l1t::PFCandidateRef& seedChargedPFCand() const { return seedChargedPFCand_; }
    const l1t::PFJetRef& seedPFJet() const { return seedPFJet_; }
    const l1t::PFCandidateRef& leadChargedPFCand() const { return leadChargedPFCand_; }

    const l1t::PFCandidateRefVector& signalAllL1PFCandidates() const { return signalAllL1PFCandidates_; }
    const l1t::PFCandidateRefVector& signalChargedHadrons() const { return signalChargedHadrons_; }
    const l1t::PFCandidateRefVector& signalElectrons() const { return signalElectrons_; }
    const l1t::PFCandidateRefVector& signalNeutralHadrons() const { return signalNeutralHadrons_; }
    const l1t::PFCandidateRefVector& signalPhotons() const { return signalPhotons_; }
    const l1t::PFCandidateRefVector& signalMuons() const { return signalMuons_; }

    const l1t::PFCandidateRefVector& stripAllL1PFCandidates() const { return stripAllL1PFCandidates_; }
    const l1t::PFCandidateRefVector& stripElectrons() const { return stripElectrons_; }
    const l1t::PFCandidateRefVector& stripPhotons() const { return stripPhotons_; }

    const l1t::PFCandidateRefVector& isoAllL1PFCandidates() const { return isoAllL1PFCandidates_; }
    const l1t::PFCandidateRefVector& isoChargedHadrons() const { return isoChargedHadrons_; }
    const l1t::PFCandidateRefVector& isoElectrons() const { return isoElectrons_; }
    const l1t::PFCandidateRefVector& isoNeutralHadrons() const { return isoNeutralHadrons_; }
    const l1t::PFCandidateRefVector& isoPhotons() const { return isoPhotons_; }
    const l1t::PFCandidateRefVector& isoMuons() const { return isoMuons_; }

    const l1t::PFCandidateRefVector& sumAllL1PFCandidates() const { return sumAllL1PFCandidates_; }
    const l1t::PFCandidateRefVector& sumChargedHadrons() const { return sumChargedHadrons_; }
    const l1t::PFCandidateRefVector& sumElectrons() const { return sumElectrons_; }
    const l1t::PFCandidateRefVector& sumNeutralHadrons() const { return sumNeutralHadrons_; }
    const l1t::PFCandidateRefVector& sumPhotons() const { return sumPhotons_; }
    const l1t::PFCandidateRefVector& sumMuons() const { return sumMuons_; }

    const l1t::TkPrimaryVertexRef& primaryVertex() const { return primaryVertex_; }

    enum Kind { kUndefined, kOneProng0Pi0, kOneProng1Pi0, kThreeProng0Pi0, kThreeProng1Pi0 };
    Kind tauType() const { return tauType_; }

    const reco::Particle::LorentzVector& strip_p4() const { return strip_p4_; }

    float sumAllL1PFCandidates_pt() const { return sumAllL1PFCandidates_pt_; }
    float signalConeSize() const { return signalConeSize_; }
    float isolationConeSize() const { return signalConeSize_; }

    float sumChargedIso() const { return sumChargedIso_; }
    float sumNeutralIso() const { return sumNeutralIso_; }
    float sumCombinedIso() const { return sumCombinedIso_; }
    float sumChargedIsoPileup() const { return sumChargedIsoPileup_; }
    float rhoCorr() const { return rhoCorr_; }

    bool passTightIso() const { return passTightIso_; }
    bool passMediumIso() const { return passMediumIso_; }
    bool passLooseIso() const { return passLooseIso_; }
    bool passVLooseIso() const { return passVLooseIso_; }

    bool passTightRelIso() const { return passTightRelIso_; }
    bool passMediumRelIso() const { return passMediumRelIso_; }
    bool passLooseRelIso() const { return passLooseRelIso_; }
    bool passVLooseRelIso() const { return passVLooseRelIso_; }

  private:
    l1t::PFCandidateRef seedChargedPFCand_;
    l1t::PFJetRef seedPFJet_;
    l1t::PFCandidateRef leadChargedPFCand_;

    l1t::PFCandidateRefVector signalAllL1PFCandidates_;
    l1t::PFCandidateRefVector signalChargedHadrons_;
    l1t::PFCandidateRefVector signalElectrons_;
    l1t::PFCandidateRefVector signalNeutralHadrons_;
    l1t::PFCandidateRefVector signalPhotons_;
    l1t::PFCandidateRefVector signalMuons_;

    l1t::PFCandidateRefVector stripAllL1PFCandidates_;
    l1t::PFCandidateRefVector stripElectrons_;
    l1t::PFCandidateRefVector stripPhotons_;

    l1t::PFCandidateRefVector isoAllL1PFCandidates_;
    l1t::PFCandidateRefVector isoChargedHadrons_;
    l1t::PFCandidateRefVector isoElectrons_;
    l1t::PFCandidateRefVector isoNeutralHadrons_;
    l1t::PFCandidateRefVector isoPhotons_;
    l1t::PFCandidateRefVector isoMuons_;

    l1t::PFCandidateRefVector sumAllL1PFCandidates_;
    l1t::PFCandidateRefVector sumChargedHadrons_;
    l1t::PFCandidateRefVector sumElectrons_;
    l1t::PFCandidateRefVector sumNeutralHadrons_;
    l1t::PFCandidateRefVector sumPhotons_;
    l1t::PFCandidateRefVector sumMuons_;

    l1t::TkPrimaryVertexRef primaryVertex_;
    Kind tauType_;

    reco::Particle::LorentzVector strip_p4_;

    float sumAllL1PFCandidates_pt_;
    float signalConeSize_;
    float isolationConeSize_;

    float sumChargedIso_;
    float sumNeutralIso_;
    float sumCombinedIso_;
    float sumChargedIsoPileup_;  // charged PFCands failing dz cut (maybe useful to correct neutral isolation for pile-up contributions by applying delta-beta corrections)
    float rhoCorr_;              // rho correction (maybe useful for applying pile-up corrections to neutral isolation)

    bool passTightIso_;
    bool passMediumIso_;
    bool passLooseIso_;
    bool passVLooseIso_;

    bool passTightRelIso_;
    bool passMediumRelIso_;
    bool passLooseRelIso_;
    bool passVLooseRelIso_;
  };

  void printPFCand(ostream& os, const l1t::PFCandidate& l1PFCand, const l1t::TkPrimaryVertexRef& primaryVertex);
  void printPFCand(ostream& os, const l1t::PFCandidate& l1PFCand, float primaryVertex_z);
}  // namespace l1t

/// print to stream
std::ostream& operator<<(std::ostream& os, const l1t::HPSPFTau& l1PFTau);

#endif
