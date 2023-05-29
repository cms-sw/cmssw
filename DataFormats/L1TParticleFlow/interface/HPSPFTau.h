#ifndef DataFormats_L1TParticleFlow_HPSPFTau_H
#define DataFormats_L1TParticleFlow_HPSPFTau_H

#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"  // l1t::PFCandidate, l1t::PFCandidateRef, l1t::PFCandidateRefVector
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"  // reco::LeafCandidate
#include "DataFormats/Candidate/interface/Particle.h"       // reco::Particle::LorentzVector
#include "DataFormats/L1Trigger/interface/VertexWord.h"

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
    bool isJetSeeded() const { return seedJet_.isNonnull(); }

    const l1t::PFCandidateRef& seedChargedPFCand() const { return seedChargedPFCand_; }
    const reco::CaloJetRef& seedJet() const { return seedJet_; }
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

    const l1t::VertexWordRef& primaryVertex() const { return primaryVertex_; }

    enum Kind { kUndefined, kOneProng0Pi0, kOneProng1Pi0, kThreeProng0Pi0, kThreeProng1Pi0 };
    Kind tauType() const { return tauType_; }

    const reco::Particle::LorentzVector& stripP4() const { return stripP4_; }

    float sumAllL1PFCandidatesPt() const { return sumAllL1PFCandidatesPt_; }
    float signalConeSize() const { return signalConeSize_; }
    float isolationConeSize() const { return signalConeSize_; }

    float sumChargedIso() const { return sumChargedIso_; }
    float sumNeutralIso() const { return sumNeutralIso_; }
    float sumCombinedIso() const { return sumCombinedIso_; }
    float sumChargedIsoPileup() const { return sumChargedIsoPileup_; }
    float z() const { return z_; }

    bool passTightIso() const { return passTightIso_; }
    bool passMediumIso() const { return passMediumIso_; }
    bool passLooseIso() const { return passLooseIso_; }
    bool passVLooseIso() const { return passVLooseIso_; }

    bool passTightRelIso() const { return passTightRelIso_; }
    bool passMediumRelIso() const { return passMediumRelIso_; }
    bool passLooseRelIso() const { return passLooseRelIso_; }
    bool passVLooseRelIso() const { return passVLooseRelIso_; }

    void setSeedChargedPFCand(l1t::PFCandidateRef seedChargedPFCand) { seedChargedPFCand_ = seedChargedPFCand; }
    void setSeedJet(reco::CaloJetRef seedJet) { seedJet_ = seedJet; }
    void setLeadChargedPFCand(l1t::PFCandidateRef leadChargedPFCand) { leadChargedPFCand_ = leadChargedPFCand; }

    void setSignalAllL1PFCandidates(l1t::PFCandidateRefVector signalAllL1PFCandidates) {
      signalAllL1PFCandidates_ = signalAllL1PFCandidates;
    }
    void setSignalChargedHadrons(l1t::PFCandidateRefVector signalChargedHadrons) {
      signalChargedHadrons_ = signalChargedHadrons;
    }
    void setSignalElectrons(l1t::PFCandidateRefVector signalElectrons) { signalElectrons_ = signalElectrons; }
    void setSignalNeutralHadrons(l1t::PFCandidateRefVector signalNeutralHadrons) {
      signalNeutralHadrons_ = signalNeutralHadrons;
    }
    void setSignalPhotons(l1t::PFCandidateRefVector signalPhotons) { signalPhotons_ = signalPhotons; }
    void setSignalMuons(l1t::PFCandidateRefVector signalMuons) { signalMuons_ = signalMuons; }

    void setStripAllL1PFCandidates(l1t::PFCandidateRefVector stripAllL1PFCandidates) {
      stripAllL1PFCandidates_ = stripAllL1PFCandidates;
    }
    void setStripElectrons(l1t::PFCandidateRefVector stripElectrons) { stripElectrons_ = stripElectrons; }
    void setStripPhotons(l1t::PFCandidateRefVector stripPhotons) { stripPhotons_ = stripPhotons; }

    void setIsoAllL1PFCandidates(l1t::PFCandidateRefVector isoAllL1PFCandidates) {
      isoAllL1PFCandidates_ = isoAllL1PFCandidates;
    }
    void setIsoChargedHadrons(l1t::PFCandidateRefVector isoChargedHadrons) { isoChargedHadrons_ = isoChargedHadrons; }
    void setIsoElectrons(l1t::PFCandidateRefVector isoElectrons) { isoElectrons_ = isoElectrons; }
    void setIsoNeutralHadrons(l1t::PFCandidateRefVector isoNeutralHadrons) { isoNeutralHadrons_ = isoNeutralHadrons; }
    void setIsoPhotons(l1t::PFCandidateRefVector isoPhotons) { isoPhotons_ = isoPhotons; }
    void setIsoMuons(l1t::PFCandidateRefVector isoMuons) { isoMuons_ = isoMuons; }

    void setSumAllL1PFCandidates(l1t::PFCandidateRefVector sumAllL1PFCandidates) {
      sumAllL1PFCandidates_ = sumAllL1PFCandidates;
    }
    void setSumChargedHadrons(l1t::PFCandidateRefVector sumChargedHadrons) { sumChargedHadrons_ = sumChargedHadrons; }
    void setSumElectrons(l1t::PFCandidateRefVector sumElectrons) { sumElectrons_ = sumElectrons; }
    void setSumNeutralHadrons(l1t::PFCandidateRefVector sumNeutralHadrons) { sumNeutralHadrons_ = sumNeutralHadrons; }
    void setSumPhotons(l1t::PFCandidateRefVector sumPhotons) { sumPhotons_ = sumPhotons; }
    void setSumMuons(l1t::PFCandidateRefVector sumMuons) { sumMuons_ = sumMuons; }

    void setPrimaryVertex(l1t::VertexWordRef primaryVertex) { primaryVertex_ = primaryVertex; }

    void setTauType(Kind tauType) { tauType_ = tauType; }

    void setStripP4(reco::Particle::LorentzVector& stripP4) { stripP4_ = stripP4; }

    void setSumAllL1PFCandidatesPt(float sumAllL1PFCandidatesPt) { sumAllL1PFCandidatesPt_ = sumAllL1PFCandidatesPt; }
    void setSignalConeSize(float signalConeSize) { signalConeSize_ = signalConeSize; }
    void setisolationConeSize(float isolationConeSize) { signalConeSize_ = isolationConeSize; }

    void setSumChargedIso(float sumChargedIso) { sumChargedIso_ = sumChargedIso; }
    void setSumNeutralIso(float sumNeutralIso) { sumNeutralIso_ = sumNeutralIso; }
    void setSumCombinedIso(float sumCombinedIso) { sumCombinedIso_ = sumCombinedIso; }
    void setSumChargedIsoPileup(float sumChargedIsoPileup) { sumChargedIsoPileup_ = sumChargedIsoPileup; }
    void setZ(float Z) { z_ = Z; }

    void setPassTightIso(bool passTightIso) { passTightIso_ = passTightIso; }
    void setPassMediumIso(bool passMediumIso) { passMediumIso_ = passMediumIso; }
    void setPassLooseIso(bool passLooseIso) { passLooseIso_ = passLooseIso; }
    void setPassVLooseIso(bool passVLooseIso) { passVLooseIso_ = passVLooseIso; }

    void setPassTightRelIso(bool passTightRelIso) { passTightRelIso_ = passTightRelIso; }
    void setPassMediumRelIso(bool passMediumRelIso) { passMediumRelIso_ = passMediumRelIso; }
    void setPassLooseRelIso(bool passLooseRelIso) { passLooseRelIso_ = passLooseRelIso; }
    void setPassVLooseRelIso(bool passVLooseRelIso) { passVLooseRelIso_ = passVLooseRelIso; }

  private:
    l1t::PFCandidateRef seedChargedPFCand_;
    reco::CaloJetRef seedJet_;
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

    l1t::VertexWordRef primaryVertex_;
    Kind tauType_;

    reco::Particle::LorentzVector stripP4_;

    float sumAllL1PFCandidatesPt_;
    float signalConeSize_;
    float isolationConeSize_;

    float sumChargedIso_;
    float sumNeutralIso_;
    float sumCombinedIso_;
    float sumChargedIsoPileup_;  // charged PFCands failing dz cut (maybe useful to correct neutral isolation for pile-up contributions by applying delta-beta corrections)

    bool passTightIso_;
    bool passMediumIso_;
    bool passLooseIso_;
    bool passVLooseIso_;

    bool passTightRelIso_;
    bool passMediumRelIso_;
    bool passLooseRelIso_;
    bool passVLooseRelIso_;

    float z_;
  };

}  // namespace l1t

/// print to stream
std::ostream& operator<<(std::ostream& os, const l1t::HPSPFTau& l1PFTau);

void printPFCand(ostream& os, const l1t::PFCandidate& l1PFCand, const l1t::VertexWordRef& primaryVertex);
void printPFCand(ostream& os, const l1t::PFCandidate& l1PFCand, float primaryVertexZ);

#endif
