#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "PhysicsTools/PatUtils/interface/MiniIsolation.h"

namespace pat{

// Computes MiniIsolation given a 4-vector of the object in question
// and a collection of packed PF cands. Computed as a sum of PFCand pT
// inside a cone of radius dR = max(mindir, min(maxdr, kt_scale/pT))
// Excludes PFCands inside of "deadcone" radius.
// For nh, ph, pu, only include particles with pT > ptthresh
// Some documentation can be found here: https://hypernews.cern.ch/HyperNews/CMS/get/susy/1991.html
PFIsolation getMiniPFIsolation(const pat::PackedCandidateCollection *pfcands,
                                 const math::XYZTLorentzVector &p4, float mindr, float maxdr,
                                 float kt_scale, float ptthresh, float deadcone_ch,
                                 float deadcone_pu, float deadcone_ph, float deadcone_nh,
                                 float dZ_cut)
{
    
    float chiso=0, nhiso=0, phiso=0, puiso=0;
    float drcut = std::max(mindr, std::min(maxdr, float(kt_scale/p4.pt())));
    for(auto const & pc : *pfcands){
        float dr = deltaR(p4, pc.p4());
        if(dr>drcut)
            continue;
        float pt = pc.p4().pt();
        int id = pc.pdgId();
        if(std::abs(id)==211){
            bool fromPV = (pc.fromPV()>1 || fabs(pc.dz()) < dZ_cut);
            if(fromPV && dr > deadcone_ch){
                // if charged hadron and from primary vertex, add to charged hadron isolation
                chiso += pt;
            }else if(!fromPV && pt > ptthresh && dr > deadcone_pu){
                // if charged hadron and NOT from primary vertex, add to pileup isolation
                puiso += pt;
            }
        }
        // if neutral hadron, add to neutral hadron isolation
        if(std::abs(id)==130 && pt>ptthresh && dr>deadcone_nh)
            nhiso += pt;
        // if photon, add to photon isolation
        if(std::abs(id)==22 && pt>ptthresh && dr>deadcone_ph)
            phiso += pt;
                
    }
                      
    return pat::PFIsolation(chiso, nhiso, phiso, puiso);

}

}
