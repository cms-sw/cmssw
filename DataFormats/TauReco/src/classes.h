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

#include <vector>
#include <map>

namespace {
  struct dictionary {
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

    reco::CaloTauDiscriminatorAgainstElectronBase               calotde_b;
    reco::CaloTauDiscriminatorAgainstElectron                   calotde_o;
    reco::CaloTauDiscriminatorAgainstElectronRef                calotde_r;
    reco::CaloTauDiscriminatorAgainstElectronRefProd            calotde_rp;
    reco::CaloTauDiscriminatorAgainstElectronRefVector          calotde_rv;
    edm::Wrapper<reco::CaloTauDiscriminatorAgainstElectron>     calotde_w;

    std::pair<reco::CaloTauRef, int>                            calotd_p;
    std::vector<std::pair<reco::CaloTauRef, int> >              calotd_v;

    reco::PFTauDiscriminatorByIsolationBase                     pftdi_b;
    reco::PFTauDiscriminatorByIsolation                         pftdi_o;
    reco::PFTauDiscriminatorByIsolationRef                      pftdi_r;
    reco::PFTauDiscriminatorByIsolationRefProd                  pftdi_rp;
    reco::PFTauDiscriminatorByIsolationRefVector                pftdi_rv;
    edm::Wrapper<reco::PFTauDiscriminatorByIsolation>           pftdi_w;

    std::pair<reco::PFTauRef, int>                              pftd_p;
    std::vector<std::pair<reco::PFTauRef, int> >                pftd_v;


    reco::PFTauDiscriminatorBase                     pftdiscr_b;
    reco::PFTauDiscriminator                         pftdiscr_o;
    reco::PFTauDiscriminatorRef                      pftdiscr_r;
    reco::PFTauDiscriminatorRefProd                  pftdiscr_rp;
    reco::PFTauDiscriminatorRefVector                pftdiscr_rv;
    edm::Wrapper<reco::PFTauDiscriminator>           pftdiscr_w;

    std::pair<reco::PFTauRef, float>                              pftdiscr_p;
    std::vector<std::pair<reco::PFTauRef, float> >                pftdiscr_v;

    reco::JetPiZeroAssociationBase                     jetPiZeroAssoc_b;
    reco::JetPiZeroAssociation                         jetPiZeroAssoc_o;
    reco::JetPiZeroAssociationRef                      jetPiZeroAssoc_r;
    reco::JetPiZeroAssociationRefProd                  jetPiZeroAssoc_rp;
    reco::JetPiZeroAssociationRefVector                jetPiZeroAssoc_rv;
    edm::Wrapper<reco::JetPiZeroAssociation>           jetPiZeroAssoc_w;

    std::pair<reco::PFJetRef, std::vector<reco::RecoTauPiZero> >                              jetPiZeroAssoc_p;
    std::vector<std::pair<reco::PFJetRef, std::vector<reco::RecoTauPiZero> > >                jetPiZeroAssoc_v;

    std::vector<std::vector<reco::RecoTauPiZero> >                jetPiZeroAssoc_v_v;

    reco::PFTauDecayModeAssociation                         pftdecaymodeass_o;
    reco::PFTauDecayModeAssociationRef                      pftdecaymodeass_r;
    reco::PFTauDecayModeAssociationRefProd                  pftdecaymodeass_rp;
    reco::PFTauDecayModeAssociationRefVector                pftdecaymodeass_rv;
    edm::Wrapper<reco::PFTauDecayModeAssociation>           pftdecaymodeass_w;

    std::pair<reco::PFTauRef, reco::PFTauDecayMode>                              pftdecaymodeass_p;
    std::vector<std::pair<reco::PFTauRef, reco::PFTauDecayMode> >                pftdecaymodeass_v;
    std::pair<reco::CaloTauRef, float>                              calodiscr_p;
    std::vector<std::pair<reco::CaloTauRef, float> >                calodiscr_v;

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
