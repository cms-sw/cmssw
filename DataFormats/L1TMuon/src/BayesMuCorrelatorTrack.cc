/*
 * BayesMuCorrelatorTrack.cc
 *
 *  Created on: Mar 15, 2019
 *      Author: Author: Karol Bunkowski kbunkow@cern.ch
 */

#include "DataFormats/L1TMuon/interface/BayesMuCorrelatorTrack.h"

namespace l1t {
BayesMuCorrelatorTrack::BayesMuCorrelatorTrack(): L1Candidate() {

}

BayesMuCorrelatorTrack::BayesMuCorrelatorTrack(const LorentzVector& p4): L1Candidate(p4) {

}

BayesMuCorrelatorTrack::BayesMuCorrelatorTrack(const edm::Ptr< L1TTTrackType > ttTrackPtr):
    L1Candidate(LorentzVector(ttTrackPtr->getMomentum().x(),
                              ttTrackPtr->getMomentum().y(),
                              ttTrackPtr->getMomentum().z(),
                              ttTrackPtr->getMomentum().mag()) ),
    ttTrackPtr(ttTrackPtr)
{

}

BayesMuCorrelatorTrack::~BayesMuCorrelatorTrack() {

}

} //end of namespace l1t
