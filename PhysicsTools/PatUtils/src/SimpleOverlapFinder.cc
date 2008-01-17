#include "PhysicsTools/PatUtils/interface/SimpleOverlapFinder.h"

#include "DataFormats/Math/interface/deltaR.h"
#include <algorithm>

std::auto_ptr< pat::SimpleOverlapFinder::Overlaps >
pat::SimpleOverlapFinder::find(
                const std::vector< const reco::Candidate * > &toClean,
                const edm::View<reco::Candidate> &theOthers) const 
{
    std::vector< const reco::Candidate * > tmp;
    for ( edm::View<reco::Candidate>::const_iterator it = theOthers.begin(), ed = theOthers.end();
            it != ed; ++it) {
        tmp.push_back(&*it);
    }
    return find(toClean, tmp);
                    
}

std::auto_ptr< pat::SimpleOverlapFinder::Overlaps >
pat::SimpleOverlapFinder::find(
                const std::vector< const reco::Candidate * > &toClean,
                const std::vector< const reco::Candidate * > &theOthers) const  
{
    using namespace std;

    auto_ptr< Overlaps > ret(new Overlaps());
    
    typedef std::vector< const reco::Candidate * >::const_iterator IT;
    IT ed2 = theOthers.end();
    int idx = 0;
    for ( IT it = toClean.begin(), ed = toClean.end(); it != ed; ++it, ++idx) {
        double dr2min = deltaR2_;
        const reco::Candidate *match = 0;
        for (IT it2 = theOthers.begin(); it2 != ed2; ++it2) {
            double dr = deltaR2(**it, **it2);
            if (dr < dr2min) {
                dr2min = dr;
                match = *it2;
            }        
        }
        if (match != 0) {
            ret->push_back(Overlap(idx,match));
        }
    }

    return ret;
}
