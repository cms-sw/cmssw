#include "PhysicsTools/PatUtils/interface/DuplicatedElectronRemover.h"

#include <algorithm>

/* --- Original comment from TQAF follows ----
 * it is possible that there are multiple electron objects in the collection that correspond to the same
 * real physics object - a supercluster with two tracks reconstructed to it, or a track that points to two different SC
 *  (i would guess the latter doesn't actually happen).
 * NB triplicates also appear in the electron collection provided by egamma group, it is necessary to handle those correctly   
 */

std::auto_ptr< std::vector<size_t> >
pat::DuplicatedElectronRemover::duplicatesToRemove(const std::vector<reco::PixelMatchGsfElectron> &electrons) 
{
    using namespace std;

    vector<bool> bad;
    fill(bad.begin(), bad.end(), false);

    size_t size = electrons.size();
    for (size_t ie = 0; ie < size; ++ie) {
        if (bad[ie]) continue; // if already marked bad

        reco::GsfTrackRef thistrack  = electrons[ie].gsfTrack();
        reco::SuperClusterRef thissc = electrons[ie].superCluster();

        for (size_t je = ie+1; je < size; ++je) {
            if (bad[je]) continue; // if already marked bad

            if ( ( thistrack == electrons[je].gsfTrack()) ||
                    (thissc  == electrons[je].superCluster()) ) {
                // we have a match, arbitrate and mark one for removal
                // keep the one with E/P closer to unity
                float diff1 = fabs(electrons[ie].eSuperClusterOverP()-1);
                float diff2 = fabs(electrons[je].eSuperClusterOverP()-1);

                if (diff1<diff2) {
                    bad[je] = true;
                } else {
                    bad[ie] = true;
                }
            }
        }
    }

    auto_ptr< vector<size_t> > ret(new vector<size_t>());

    for (size_t i = 0; i < size; ++i) {
        if (bad[i]) ret->push_back(i);
    }

    return ret;
}
