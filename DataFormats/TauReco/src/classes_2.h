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
#include "DataFormats/Common/interface/RefToPtr.h"
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
  struct dictionary2 {
    std::vector<reco::PFTau>                                    pft_v;
    edm::Wrapper<std::vector<reco::PFTau> >                     pft_w;
    edm::Ref<std::vector<reco::PFTau> >                         pft_r;
    edm::RefProd<std::vector<reco::PFTau> >                     pft_rp;
    edm::RefVector<std::vector<reco::PFTau> >                   pft_rv;
    edm::Wrapper<edm::RefVector<std::vector<reco::PFTau> > >         pft_rvw;
    edm::reftobase::Holder<reco::BaseTau,reco::PFTauRef>        pft_rb;

    edm::View<reco::PFTau>  jv;
    edm::RefToBaseProd<reco::PFTau> jrtbp;

    edm::RefToBase<reco::PFTau>  rtbj;
    edm::reftobase::IndirectHolder<reco::PFTau> ihj;
    edm::reftobase::Holder<reco::PFTau, reco::PFTauRef> hcj;
    edm::reftobase::RefHolder<reco::PFTauRef> rhtj;
    edm::RefToBaseVector<reco::PFTau> jrtbv;
    edm::Wrapper<edm::RefToBaseVector<reco::PFTau> > jrtbv_w;
    edm::reftobase::BaseVectorHolder<reco::PFTau> * bvhj_p;    // pointer since it's pure virtual

    std::vector<reco::PFTauDecayMode>                                           pftdm_v;
    edm::Wrapper<std::vector<reco::PFTauDecayMode> >                            pftdm_w;
    edm::Ref<std::vector<reco::PFTauDecayMode> >                                pftdm_r;
    edm::RefProd<std::vector<reco::PFTauDecayMode> >                            pftdm_rp;
    edm::RefVector<std::vector<reco::PFTauDecayMode> >                          pftdm_rv;
    edm::reftobase::Holder<reco::CompositeCandidate,reco::PFTauDecayModeRef>    pftdm_rb;
    edm::Association<std::vector<reco::PFTauDecayMode> >                        pftdm_assoc_v;
    edm::Association<std::vector<reco::PFTau> >                                 pftau_assoc_v; // used for matching
    edm::Wrapper<edm::Association<std::vector<reco::PFTauDecayMode> > >         pftdm_assoc_v_wrapper;
    edm::Wrapper<edm::Association<std::vector<reco::PFTau> > >                  pftau_assoc_v_wrapper;

    std::vector<reco::RecoTauPiZero>                                           recoTauPiZero_v;
    edm::Wrapper<std::vector<reco::RecoTauPiZero> >                            recoTauPiZero_w;
    edm::Ref<std::vector<reco::RecoTauPiZero> >                                recoTauPiZero_r;
    edm::RefProd<std::vector<reco::RecoTauPiZero> >                            recoTauPiZero_rp;
    edm::RefVector<std::vector<reco::RecoTauPiZero> >                          recoTauPiZero_rv;
    edm::reftobase::Holder<reco::CompositePtrCandidate, reco::RecoTauPiZeroRef>    recoTauPiZero_rb;

    std::vector<reco::PFRecoTauChargedHadron>                                           pfrecoTauChH_v;
    edm::Wrapper<std::vector<reco::PFRecoTauChargedHadron> >                            pfrecoTauChH_w;
    edm::Ref<std::vector<reco::PFRecoTauChargedHadron> >                                pfrecoTauChH_r;
    edm::RefProd<std::vector<reco::PFRecoTauChargedHadron> >                            pfrecoTauChH_rp;
    edm::RefVector<std::vector<reco::PFRecoTauChargedHadron> >                          pfrecoTauChH_rv;
    edm::reftobase::Holder<reco::CompositePtrCandidate, reco::PFRecoTauChargedHadronRef>        pfrecoTauChH_rb;


    reco::CaloTauDiscriminatorByIsolationBase                   calotdi_b;
    reco::CaloTauDiscriminatorByIsolation                       calotdi_o;
    reco::CaloTauDiscriminatorByIsolationRef                    calotdi_r;
    reco::CaloTauDiscriminatorByIsolationRefProd                calotdi_rp;
    reco::CaloTauDiscriminatorByIsolationRefVector              calotdi_rv;
    edm::Wrapper<reco::CaloTauDiscriminatorByIsolation>         calotdi_w;

    reco::CaloTauDiscriminatorBase                   calodi_b;
    reco::CaloTauDiscriminator                       calodi_o;
    reco::CaloTauDiscriminatorRef                    calodi_r;
    reco::CaloTauDiscriminatorRefProd                calodi_rp;
    reco::CaloTauDiscriminatorRefVector              calodi_rv;
    edm::Wrapper<reco::CaloTauDiscriminator>         calodi_w;
  };
}



