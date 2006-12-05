#ifndef JetAlgorithms_JetRecoTypes_h
#define JetAlgorithms_JetRecoTypes_h

// Types used in Jet Reconstruction
// F.Ratnikov, UMd
// $Id: JetAlgoHelper.h,v 1.3 2006/11/17 16:18:04 tboccali Exp $

#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace JetReco {
    typedef reco::CandidateRef InputItem;
    typedef std::vector <InputItem> InputCollection;
    typedef std::vector <ProtoJet> OutputCollection;
}
#endif
