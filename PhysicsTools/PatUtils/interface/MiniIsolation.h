#ifndef MiniIsolation_h
#define MiniIsolation_h

/*

  Defines a MiniIsolation struct and a function to
  compute MiniIsolation given a 4-vector and a collection
  of packed PF candidates

*/

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"


namespace pat{

    struct MiniIsolation {
        float chiso; //charged hadrons
        float nhiso; //neutral hadrons
        float phiso; //photons
        float puiso; //pileup
    };

    typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > LorentzVector;

    MiniIsolation GetMiniPFIsolation(const pat::PackedCandidateCollection *pfcands, LorentzVector p4,
                                     float mindr=0.05, float maxdr=0.2, float kt_scale=10.0,
                                     float ptthresh=0.5, float deadcone_ch=0.0001, 
                                     float deadcone_pu=0.01, float deadcone_ph=0.01, float deadcone_nh=0.01,
                                     float dZ_cut=0.0);

}

#endif
