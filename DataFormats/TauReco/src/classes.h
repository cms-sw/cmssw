#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/TauReco/interface/BaseTauTagInfo.h"
#include "DataFormats/TauReco/interface/CaloTauTagInfo.h"
#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminatorByIsolation.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminatorAgainstElectron.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminatorByIsolation.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TauReco/interface/L2TauInfoAssociation.h"
#include "DataFormats/TauReco/interface/HLTTau.h"

#include <vector>
#include <map>

namespace {
  namespace {
    reco::L2TauIsolationInfo                                    l2iI;
    L2TauInfoAssociation                                        l2ts;
    edm::Wrapper<L2TauInfoAssociation>                          wl2ts;


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

    std::vector<reco::PFTau>                                    pft_v;
    edm::Wrapper<std::vector<reco::PFTau> >                     pft_w;
    edm::Ref<std::vector<reco::PFTau> >                         pft_r;
    edm::RefProd<std::vector<reco::PFTau> >                     pft_rp;
    edm::RefVector<std::vector<reco::PFTau> >                   pft_rv;
    edm::reftobase::Holder<reco::BaseTau,reco::PFTauRef>        pft_rb;

    reco::CaloTauDiscriminatorByIsolationBase                   calotdi_b;         
    reco::CaloTauDiscriminatorByIsolation                       calotdi_o;     
    reco::CaloTauDiscriminatorByIsolationRef                    calotdi_r;     
    reco::CaloTauDiscriminatorByIsolationRefProd                calotdi_rp;     
    reco::CaloTauDiscriminatorByIsolationRefVector              calotdi_rv;     
    edm::Wrapper<reco::CaloTauDiscriminatorByIsolation>         calotdi_w;     
    
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


    //Needed only in HLT-Open
    std::vector<reco::HLTTau>                                  ht_v;
    edm::Wrapper<std::vector<reco::HLTTau> >                   ht_w;
    edm::Ref<std::vector<reco::HLTTau> >                       ht_r;
    edm::RefProd<std::vector<reco::HLTTau> >                   ht_rp;
    edm::RefVector<std::vector<reco::HLTTau> >                 ht_rv;




  }
}
