#ifndef PhysicsTools_Heppy_IsolationComputer_h
#define PhysicsTools_Heppy_IsolationComputer_h

#include <vector>
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

namespace heppy {
class IsolationComputer {
    public:
        /// Create the calculator; optionally specify a cone for computing deltaBeta weights
        IsolationComputer(float weightCone=-1) : weightCone_(weightCone) {}

        /// Self-veto policy
        enum SelfVetoPolicy {
            selfVetoNone=0, selfVetoAll=1, selfVetoFirst=2
        };
        /// Initialize with the list of packed candidates (note: clears also all vetos)
        void setPackedCandidates(const std::vector<pat::PackedCandidate> & all, int fromPV_thresh=1, float dz_thresh=9999., float dxy_thresh=9999., bool also_leptons=false) ;


        /// veto footprint from this candidate, for the isolation of all candidates and also for calculation of neutral weights (if used)
        void addVetos(const reco::Candidate &cand) ;

        /// clear all vetos
        void clearVetos() ;

        /// Isolation from charged from the PV 
        float chargedAbsIso(const reco::Candidate &cand, float dR, float innerR=0, float threshold=0, SelfVetoPolicy selfVeto=selfVetoAll) const ;

        /// Isolation from charged from PU
        float puAbsIso(const reco::Candidate &cand, float dR, float innerR=0, float threshold=0, SelfVetoPolicy selfVeto=selfVetoAll) const ;

        /// Isolation from all neutrals (uncorrected)
        float neutralAbsIsoRaw(const reco::Candidate &cand, float dR, float innerR=0, float threshold=0, SelfVetoPolicy selfVeto=selfVetoAll) const ;

        /// Isolation from neutral hadrons (uncorrected)
        float neutralHadAbsIsoRaw(const reco::Candidate &cand, float dR, float innerR=0, float threshold=0, SelfVetoPolicy selfVeto=selfVetoAll) const ;

        /// Isolation from photons (uncorrected)
        float photonAbsIsoRaw(const reco::Candidate &cand, float dR, float innerR=0, float threshold=0, SelfVetoPolicy selfVeto=selfVetoAll) const ;

        /// Isolation from all neutrals (with weights)
        float neutralAbsIsoWeighted(const reco::Candidate &cand, float dR, float innerR=0, float threshold=0, SelfVetoPolicy selfVeto=selfVetoAll) const ;

        /// Isolation from neutral hadrons (with weights)
        float neutralHadAbsIsoWeighted(const reco::Candidate &cand, float dR, float innerR=0, float threshold=0, SelfVetoPolicy selfVeto=selfVetoAll) const ;

        /// Isolation from photons (with weights)
        float photonAbsIsoWeighted(const reco::Candidate &cand, float dR, float innerR=0, float threshold=0, SelfVetoPolicy selfVeto=selfVetoAll) const ;
    protected:
        const std::vector<pat::PackedCandidate> * allcands_;
        float weightCone_;
        // collections of objects, sorted in eta
        std::vector<const pat::PackedCandidate *> charged_, neutral_, pileup_;
        mutable std::vector<float> weights_;
        std::vector<const reco::Candidate *> vetos_;

        float isoSumRaw(const std::vector<const pat::PackedCandidate *> & cands, const reco::Candidate &cand, float dR, float innerR, float threshold, SelfVetoPolicy selfVeto, int pdgId=-1) const ;
        float isoSumNeutralsWeighted(const reco::Candidate &cand, float dR, float innerR, float threshold, SelfVetoPolicy selfVeto, int pdgId=-1) const ;
};

}

#endif
