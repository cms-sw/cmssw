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
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadronFwd.h"
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
#include "DataFormats/TauReco/interface/PFJetChargedHadronAssociation.h"
#include "DataFormats/TauReco/interface/PFTauDecayModeAssociation.h"
#include "DataFormats/TauReco/interface/L2TauInfoAssociation.h"
#include "DataFormats/TauReco/interface/HLTTau.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterAssociation.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterFwd.h"


#include <vector>
#include <map>

namespace {
  struct dictionary {

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
    std::vector<std::vector<reco::RecoTauPiZero> >                                            jetPiZeroAssoc_v_v;

    reco::PFJetChargedHadronAssociationBase            jetChargedHadronAssoc_b;
    reco::PFJetChargedHadronAssociation                jetChargedHadronAssoc_o;
    reco::PFJetChargedHadronAssociationRef             jetChargedHadronAssoc_r;
    reco::PFJetChargedHadronAssociationRefProd         jetChargedHadronAssoc_rp;
    reco::PFJetChargedHadronAssociationRefVector       jetChargedHadronAssoc_rv;
    edm::Wrapper<reco::PFJetChargedHadronAssociation>  jetChargedHadronAssoc_w;

    std::pair<reco::PFJetRef, std::vector<reco::PFRecoTauChargedHadron> >                     jetChargedHadronAssoc_p;
    std::vector<std::pair<reco::PFJetRef, std::vector<reco::PFRecoTauChargedHadron> > >       jetChargedHadronAssoc_v;
    std::vector<std::vector<reco::PFRecoTauChargedHadron> >                                   jetChargedHadronAssoc_v_v;

    reco::PFTauDecayModeAssociation                         pftdecaymodeass_o;
    reco::PFTauDecayModeAssociationRef                      pftdecaymodeass_r;
    reco::PFTauDecayModeAssociationRefProd                  pftdecaymodeass_rp;
    reco::PFTauDecayModeAssociationRefVector                pftdecaymodeass_rv;
    edm::Wrapper<reco::PFTauDecayModeAssociation>           pftdecaymodeass_w;

    std::pair<reco::PFTauRef, reco::PFTauDecayMode>                              pftdecaymodeass_p;
    std::vector<std::pair<reco::PFTauRef, reco::PFTauDecayMode> >                pftdecaymodeass_v;
    std::pair<reco::CaloTauRef, float>                              calodiscr_p;
    std::vector<std::pair<reco::CaloTauRef, float> >                calodiscr_v;

    reco::PFTauTransverseImpactParameter                                                        pftautip_o;
    std::vector<reco::PFTauTransverseImpactParameter>                                           pftautip_v;
    edm::Wrapper<std::vector<reco::PFTauTransverseImpactParameter> >                            pftautip_w;
    edm::Ref<std::vector<reco::PFTauTransverseImpactParameter> >                                pftautip_r;
    edm::RefProd<std::vector<reco::PFTauTransverseImpactParameter> >                            pftautip_rp;
    edm::RefVector<std::vector<reco::PFTauTransverseImpactParameter> >                          pftautip_rv;
    edm::reftobase::Holder<reco::PFTauTransverseImpactParameter,reco::PFTauTransverseImpactParameterRef>    pftautip_rb;
    edm::Association<std::vector<reco::PFTauTransverseImpactParameter> >                        pftautip_assoc_v;
    edm::Wrapper<edm::Association<std::vector<reco::PFTauTransverseImpactParameter> > >         pftautip_assoc_v_wrapper;

    std::vector<reco::VertexRef>                                                          pftauvertex_o;
    edm::Wrapper<std::vector<reco::VertexRef> >                                           pftauvertex_w;
    edm::Ref<std::vector<reco::VertexRef> >                                               pftauvertex_r;
    edm::RefProd<std::vector<reco::VertexRef> >                                           pftauvertex_rp;
    edm::RefVector<std::vector<reco::VertexRef> >                                         pftauvertex_rv;
    edm::reftobase::Holder<reco::VertexRef,edm::Ref<std::vector<reco::VertexRef> > >      pftauvertex_rb;
    edm::Association<std::vector<reco::VertexRef> >                                       pftauvertex_assoc_v;
    edm::Wrapper<edm::Association<std::vector<reco::VertexRef> > >                        pftauvertex_assoc_v_wrapper;

    std::vector<std::vector<reco::VertexRef> >                                                                      pftauvertexv_v;
    edm::Wrapper<std::vector<std::vector<reco::VertexRef> > >                                                       pftauvertexv_w;
    edm::Ref<std::vector<std::vector<reco::VertexRef> > >                                                           pftauvertexv_r;
    edm::RefProd<std::vector<std::vector<reco::VertexRef> > >                                                       pftauvertexv_rp;
    edm::RefVector<std::vector<std::vector<reco::VertexRef> > >                                                     pftauvertexv_rv;
    edm::reftobase::Holder<std::vector<reco::VertexRef>, edm::Ref<std::vector<std::vector<reco::VertexRef> > > >    pftauvertexv_rb;
    edm::Association<std::vector<std::vector<reco::VertexRef> > >                                                   pftauvertexv_assoc_v;
    edm::Wrapper<edm::Association<std::vector<std::vector<reco::VertexRef> > > >                                    pftauvertexv_assoc_v_wrapper;

    reco::PFTauTIPAssociation                         pftautipass_o;
    reco::PFTauTIPAssociationRef                      pftautipass_r;
    reco::PFTauTIPAssociationRefProd                  pftautipass_rp;
    reco::PFTauTIPAssociationRefVector                pftautipass_rv;
    edm::Wrapper<reco::PFTauTIPAssociation>           pftautipass_w;

    std::pair<reco::PFTauRef, std::vector<reco::PFTauTransverseImpactParameter> >                pftaupairtip_o;
    std::vector<std::pair<reco::PFTauRef, std::vector<reco::PFTauTransverseImpactParameter> > >  pftaupairtip_v;

    reco::PFTauVertexAssociation                      pftauvertexass_o;
    reco::PFTauVertexAssociationRef                   pftauvertexass_r;
    reco::PFTauVertexAssociationRefProd               pftauvertexass_rp;
    reco::PFTauVertexAssociationRefVector             pftauvertexass_rv;
    edm::Wrapper<reco::PFTauVertexAssociation>        pftauvertexass_w;
    std::pair<reco::PFTauRef, std::vector<reco::VertexRef> >                 pftaupairvertex_o;
    std::vector<std::pair<reco::PFTauRef, std::vector<reco::VertexRef> > >   pftaupairvertex_v;

    reco::PFTauVertexVAssociation                     pftauvertexvass_o;
    reco::PFTauVertexVAssociationRef                  pftauvertexvass_r;
    reco::PFTauVertexVAssociationRefProd              pftauvertexvass_rp;
    reco::PFTauVertexVAssociationRefVector            pftauvertexvass_rv;
    edm::Wrapper<reco::PFTauVertexVAssociation>       pftauvertexvass_w;
    std::pair<reco::PFTauRef, std::vector<std::vector<reco::VertexRef> > >                pftaupairvertexv_o;
    std::vector<std::pair<reco::PFTauRef, std::vector<std::vector<reco::VertexRef> > > >  pftaupairvertexv_v;


  };
}
