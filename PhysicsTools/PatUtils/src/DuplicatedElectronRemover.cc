#include "PhysicsTools/PatUtils/interface/DuplicatedElectronRemover.h"

#include <algorithm>


std::auto_ptr< std::vector<size_t> > 
pat::DuplicatedElectronRemover::duplicatesToRemove(const std::vector<reco::GsfElectron> &electrons) const {
    return duplicatesToRemove< std::vector<reco::GsfElectron> >(electrons);
}

std::auto_ptr< std::vector<size_t> > 
pat::DuplicatedElectronRemover::duplicatesToRemove(const edm::View<reco::GsfElectron>   &electrons) const {
    return duplicatesToRemove< edm::View<reco::GsfElectron> >(electrons);
}




/*
std::auto_ptr< std::vector<size_t> >
pat::DuplicatedElectronRemover::duplicatesToRemove(const std::vector<reco::GsfElectron> &electrons) 
{
    using namespace std;

    size_t size = electrons.size();

    vector<bool> bad(size, false);

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
*/
