#ifndef JetAlgorithms_JetRecoTypes_h
#define JetAlgorithms_JetRecoTypes_h

// Types used in Jet Reconstruction
// F.Ratnikov, UMd
// $Id: JetRecoTypes.h,v 1.2 2007/08/15 17:43:14 fedor Exp $

#include "DataFormats/Candidate/interface/CandidateFwd.h"
class ProtoJet;

namespace JetReco {
    typedef reco::CandidateBaseRef InputItem;
    typedef std::vector<InputItem> InputCollection;
    typedef std::vector <ProtoJet> OutputCollection;
    typedef reco::CandidateBaseRefVector JetConstituents;
}
#endif
