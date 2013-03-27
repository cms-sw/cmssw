#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/TauReco/interface/BaseTauTagInfo.h"
#include "DataFormats/TauReco/interface/CaloTauTagInfo.h"
#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDecayMode.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/TauReco/interface/JetPiZeroAssociation.h"
#include "DataFormats/TauReco/interface/PFTauDecayModeAssociation.h"
#include "DataFormats/TauReco/interface/L2TauInfoAssociation.h"
#include "DataFormats/TauReco/interface/HLTTau.h"
#include "DataFormats/Common/interface/FwdPtr.h"

#include <vector>
#include <map>

namespace {
  struct dictionary {
    //Needed only in HLT-Open
    std::vector<reco::HLTTau>                                  ht_v;
    edm::Wrapper<std::vector<reco::HLTTau> >                   ht_w;
    edm::Ref<std::vector<reco::HLTTau> >                       ht_r;
    edm::RefProd<std::vector<reco::HLTTau> >                   ht_rp;
    edm::RefVector<std::vector<reco::HLTTau> >                 ht_rv;

    edm::Ptr<reco::BaseTau>	 ptr_t;
    edm::PtrVector<reco::BaseTau>	 ptrv_t;

    edm::Ptr<reco::PFTau>     ptr_pft;
    std::vector< edm::Ptr<reco::PFTau> > ptrv_pft;
    edm::Wrapper<std::vector< edm::Ptr<reco::PFTau> > > wptrv_pft;

    edm::FwdPtr<reco::PFTau>     fwdptr_pft;
    std::vector< edm::FwdPtr<reco::PFTau> > fwdptrv_pft;
    edm::Wrapper<std::vector< edm::FwdPtr<reco::PFTau> > > wfwdptrv_pft;

    edm::FwdPtr<reco::BaseTau>     fwdptr_bt;
    std::vector< edm::FwdPtr<reco::BaseTau> > fwdptrv_bt;
    edm::Wrapper<std::vector< edm::FwdPtr<reco::BaseTau> > > wfwdptrv_bt;
  };
}
