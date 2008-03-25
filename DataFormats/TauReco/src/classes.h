#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/TauReco/interface/BaseTauTagInfo.h"
#include "DataFormats/TauReco/interface/CaloTauTagInfo.h"
#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminatorByIsolation.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminatorByIsolation.h"
#include "DataFormats/TauReco/interface/Tau.h"

#include <vector>
#include <map>

namespace {
  namespace {
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
    std::pair<reco::CaloTauRef, int>                            calotdi_p;
    std::vector<std::pair<reco::CaloTauRef, int> >              calotdi_v;
  
    reco::PFTauDiscriminatorByIsolationBase                     pftdi_b;         
    reco::PFTauDiscriminatorByIsolation                         pftdi_o;     
    reco::PFTauDiscriminatorByIsolationRef                      pftdi_r;     
    reco::PFTauDiscriminatorByIsolationRefProd                  pftdi_rp;     
    reco::PFTauDiscriminatorByIsolationRefVector                pftdi_rv;     
    edm::Wrapper<reco::PFTauDiscriminatorByIsolation>           pftdi_w;     
    std::pair<reco::PFTauRef, int>                              pftdi_p;
    std::vector<std::pair<reco::PFTauRef, int> >                pftdi_v;    
  
    std::vector<reco::Tau> v1;
    edm::Wrapper<std::vector<reco::Tau> > c1;
    edm::Ref<std::vector<reco::Tau> > r1;
    edm::RefProd<std::vector<reco::Tau> > rp1;
    edm::RefVector<std::vector<reco::Tau> > rv1;
    edm::reftobase::Holder<reco::Candidate, reco::TauRef> rb1;
  }
}
