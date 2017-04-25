#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "PhysicsTools/PatUtils/interface/MiniIsolation.h"

namespace pat{

// Computes MiniIsolation given a 4-vector of the object in question
// and a collection of packed PF cands. Computed as a sum of PFCand pT
// inside a cone of radius dR = max(mindir, min(maxdr, kt_scale/pT))
// Excludes PFCands inside of "deadcone" radius.
MiniIsolation GetMiniPFIsolation(const pat::PackedCandidateCollection *pfcands,
                                 LorentzVector p4, float mindr, float maxdr, 
                                 float kt_scale, float ptthresh, float deadcone_ch, 
                                 float deadcone_pu, float deadcone_ph, float deadcone_nh,
                                 float dZ_cut)
{
    
    pat::MiniIsolation miniiso = {0., 0., 0., 0.};
    float drcut = std::max(mindr, std::min(maxdr, float(kt_scale/p4.pt())));
    for(pat::PackedCandidateCollection::const_iterator pf_it = pfcands->begin(); pf_it != pfcands->end(); pf_it++){
        float dr = deltaR(p4, pf_it->p4());
        if(dr>drcut)
            continue;
        float pt = pf_it->p4().pt();
        int id = pf_it->pdgId();
        if(abs(id)==211){
            bool fromPV = (pf_it->fromPV()>1 || fabs(pf_it->dz()) < dZ_cut);
            if(fromPV && dr > deadcone_ch){
                // if charged hadron and from primary vertex, add to charged hadron isolation
                miniiso.chiso += pt;
            }else if(!fromPV && pt > ptthresh && dr > deadcone_pu){
                // if charged hadron and NOT from primary vertex, add to pileup isolation
                miniiso.puiso += pt;
            }
        }
        // if neutral hadron, add to neutral hadron isolation
        if(abs(id)==130 && pt>ptthresh && dr>deadcone_nh)
            miniiso.nhiso += pt;
        // if photon, add to photon isolation
        if(abs(id)==22 && pt>ptthresh && dr>deadcone_ph)
            miniiso.phiso += pt;
                
    }
                      
    return miniiso;
}

}
