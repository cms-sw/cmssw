#ifndef PhysicsTools_TagAndProbe_TagProbePairMaker_h
#define PhysicsTools_TagAndProbe_TagProbePairMaker_h

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include "TRandom2.h"

namespace tnp {

    /// a simple struct to hold tag, probe and mass
    struct TagProbePair {
        reco::CandidateBaseRef tag, probe, pair;
        float mass;
        TagProbePair() {}
        TagProbePair(const reco::CandidateBaseRef &t, const reco::CandidateBaseRef &p,
                     const reco::CandidateBaseRef &tp, float m) : tag(t), probe(p), pair(tp), mass(m) {}
    };
    typedef std::vector<TagProbePair> TagProbePairs;

    class TagProbePairMaker {
        public:
            TagProbePairMaker(const edm::ParameterSet &iConfig, edm::ConsumesCollector && iC) ;


            ~TagProbePairMaker() {delete randGen_;}
            /// fill in tghe T&P pairs for this event
            TagProbePairs run(const edm::Event &iEvent) const ;
        private:
            edm::EDGetTokenT<reco::CandidateView> srcToken_;
            enum Arbitration { None, OneProbe, BestMass, Random2, NonDuplicate, OnePair };
            Arbitration arbitration_;
            double arbitrationMass_;
            void arbitrate(TagProbePairs &pairs) const ;
	    TRandom2* randGen_;

	    // SCZ
 	    bool phiCutForTwoLeg_;
 	    void phiCutByEventNumber(TagProbePairs &pairs, int eventNumber) const ;
    };
}

#endif
