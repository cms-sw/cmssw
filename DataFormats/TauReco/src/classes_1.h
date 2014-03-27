#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/TauReco/interface/BaseTauTagInfo.h"
#include "DataFormats/TauReco/interface/CaloTauTagInfo.h"
#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDecayMode.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "DataFormats/TauReco/interface/RecoTauPiZeroFwd.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminatorByIsolation.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminator.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminatorAgainstElectron.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminatorByIsolation.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/TauReco/interface/JetPiZeroAssociation.h"
#include "DataFormats/TauReco/interface/PFTauDecayModeAssociation.h"
#include "DataFormats/TauReco/interface/L2TauInfoAssociation.h"
#include "DataFormats/TauReco/interface/HLTTau.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadronFwd.h"
#include "DataFormats/TauReco/interface/PFJetChargedHadronAssociation.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterAssociation.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterFwd.h"
#include "DataFormats/TauReco/interface/PFTau3ProngSummaryFwd.h"
#include "DataFormats/TauReco/interface/PFTau3ProngSummaryAssociation.h"

#include <vector>
#include <map>
#include "TLorentzVector.h"

namespace DataFormats_TauReco {
  struct dictionary1 {
    reco::L2TauIsolationInfo                                    l2iI;
    reco::L2TauInfoAssociation                                  l2ts;
    edm::Wrapper<reco::L2TauInfoAssociation>                    wl2ts;


    std::vector<reco::BaseTauTagInfo>                           btti_v;
    edm::Wrapper<std::vector<reco::BaseTauTagInfo> >            btti_w;
    edm::Ref<std::vector<reco::BaseTauTagInfo> >                btti_r;
    edm::RefProd<std::vector<reco::BaseTauTagInfo> >            btti_rp;
    edm::RefVector<std::vector<reco::BaseTauTagInfo> >          btti_rv;

    std::vector<reco::CaloTauTagInfo>                           calotti_v;
    edm::Wrapper<std::vector<reco::CaloTauTagInfo> >            calotti_w;
    edm::Ref<std::vector<reco::CaloTauTagInfo> >                calotti_r;
    edm::RefProd<std::vector<reco::CaloTauTagInfo> >            calotti_rp;
    edm::RefVector<std::vector<reco::CaloTauTagInfo> >          calotti_rv;

    std::vector<reco::PFTauTagInfo>                             pftti_v;
    edm::Wrapper<std::vector<reco::PFTauTagInfo> >              pftti_w;
    edm::Ref<std::vector<reco::PFTauTagInfo> >                  pftti_r;
    edm::RefProd<std::vector<reco::PFTauTagInfo> >              pftti_rp;
    edm::RefVector<std::vector<reco::PFTauTagInfo> >            pftti_rv;

    std::vector<reco::BaseTau>                                  bt_v;
    edm::Wrapper<std::vector<reco::BaseTau> >                   bt_w;
    edm::Ref<std::vector<reco::BaseTau> >                       bt_r;
    edm::RefProd<std::vector<reco::BaseTau> >                   bt_rp;
    edm::RefVector<std::vector<reco::BaseTau> >                 bt_rv;
    edm::reftobase::Holder<reco::Candidate,reco::BaseTauRef>    bt_rb;

    std::vector<reco::CaloTau>                                  calot_v;
    edm::Wrapper<std::vector<reco::CaloTau> >                   calot_w;
    edm::Ref<std::vector<reco::CaloTau> >                       calot_r;
    edm::RefProd<std::vector<reco::CaloTau> >                   calot_rp;
    edm::RefVector<std::vector<reco::CaloTau> >                 calot_rv;
    edm::reftobase::Holder<reco::BaseTau,reco::CaloTauRef>      calot_rb;
    edm::RefToBaseVector<reco::CaloTau> calot_rftbv;
    edm::Wrapper<edm::RefToBaseVector<reco::CaloTau> > calot_rftbv_w;


  };
}
