#ifndef JetAlgorithms_JetRecoTypes_h
#define JetAlgorithms_JetRecoTypes_h

// Types used in Jet Reconstruction
// F.Ratnikov, UMd
// $Id: JetRecoTypes.h,v 1.1 2006/12/05 18:38:58 fedor Exp $

#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace JetReco {
    typedef reco::CandidateRef InputItem;
    typedef std::vector <InputItem> InputCollection;
    typedef std::vector <ProtoJet> OutputCollection;
}
#endif
