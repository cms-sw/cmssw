#include <utility>
#include <vector>

#include "Rtypes.h" 
#include "Math/Cartesian3D.h" 
#include "Math/CylindricalEta3D.h" 
#include "Math/Polar3D.h" 
#include "Math/PxPyPzE4D.h" 
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"  
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h" 
#include "DataFormats/BTauReco/interface/JetCrystalsAssociation.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfo.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/EMIsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/CombinedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/TauImpactParameterInfo.h"
#include "DataFormats/BTauReco/interface/TauMassTagInfo.h"
#include "DataFormats/BTauReco/interface/JetEisolAssociation.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "DataFormats/BTauReco/interface/JetTagInfo.h"

namespace reco {
    typedef TrackTauImpactParameterAssociationCollection::map_type          TrackTauImpactParameterAssociationMapType;
    typedef TrackTauImpactParameterAssociationCollection::ref_type          TrackTauImpactParameterAssociationRefType;
    typedef TauMassTagInfo::ClusterTrackAssociationCollection::map_type     TauMassTagInfo_ClusterTrackAssociationMapType;
    typedef TauMassTagInfo::ClusterTrackAssociationCollection::ref_type     TauMassTagInfo_ClusterTrackAssociationRefType;
}

namespace {
  namespace {


    reco::SecondaryVertexTagInfo::TrackData                             sv_td;
    reco::SecondaryVertexTagInfo::VertexData                            sv_vd;
    std::vector<reco::SecondaryVertexTagInfo::VertexData>               sv_vdv;
    reco::SecondaryVertexTagInfo::IndexedTrackData                      sv_itd;
    std::vector<reco::SecondaryVertexTagInfo::IndexedTrackData>         sv_itdv;
    reco::SecondaryVertexTagInfo                                        sv;
    reco::SecondaryVertexTagInfoCollection                              sv_c;
    reco::SecondaryVertexTagInfoRef                                     sv_r;
    reco::SecondaryVertexTagInfoRefProd                                 sv_rp;
    reco::SecondaryVertexTagInfoRefVector                               sv_rv;
    edm::Wrapper<reco::SecondaryVertexTagInfoCollection>                sv_wc;

    reco::CombinedTauTagInfo                                            ct;
    reco::CombinedTauTagInfoCollection                                  ct_c;
    reco::CombinedTauTagInfoRef                                         ct_r;
    reco::CombinedTauTagInfoRefProd                                     ct_rp;
    reco::CombinedTauTagInfoRefVector                                   ct_rv;
    edm::Wrapper<reco::CombinedTauTagInfoCollection>                    ct_wc;


    reco::EMIsolatedTauTagInfo                                          em;
    reco::EMIsolatedTauTagInfoCollection                                em_c;
    reco::EMIsolatedTauTagInfoRef                                       em_r;
    reco::EMIsolatedTauTagInfoRefProd                                   em_rp;
    reco::EMIsolatedTauTagInfoRefVector                                 em_rv;
    edm::Wrapper<reco::EMIsolatedTauTagInfoCollection>                  em_wc;

    reco::IsolatedTauTagInfo                                            it;
    reco::IsolatedTauTagInfoCollection                                  it_c;
    reco::IsolatedTauTagInfoRef                                         it_r;
    reco::IsolatedTauTagInfoRefProd                                     it_rp;
    reco::IsolatedTauTagInfoRefVector                                   it_rv;
    edm::Wrapper<reco::IsolatedTauTagInfoCollection>                    it_wc;


    reco::SoftLeptonProperties                                          slp;
    std::pair<edm::RefToBase<reco::Track>, reco::SoftLeptonProperties>  slp_p;
    reco::SoftLeptonTagInfo::LeptonMap                                  slp_m;

    reco::SoftLeptonTagInfo                                             sl;
    reco::SoftLeptonTagInfoCollection                                   sl_c;
    reco::SoftLeptonTagInfoRef                                          sl_r;
    reco::SoftLeptonTagInfoRefProd                                      sl_rp;
    reco::SoftLeptonTagInfoRefVector                                    sl_rv;
    edm::Wrapper<reco::SoftLeptonTagInfoCollection>                     sl_wc;

    std::pair< reco::btau::TaggingVariableName, float >                 ptt1;
    std::vector<std::pair<reco::btau::TaggingVariableName,float> >      vptt1;
    reco::TaggingVariableList                                           tvl;
    reco::TaggingVariableListCollection                                 tvl_c;
    reco::TaggingVariableListRef                                        tvl_r;
    reco::TaggingVariableListRefProd                                    tvl_rp;
    reco::TaggingVariableListRefVector                                  tvl_rv;
    edm::Wrapper<reco::TaggingVariableListCollection>                   tvl_wc;

    reco::TrackTauImpactParameterAssociation                            ttip;
    reco::TrackTauImpactParameterAssociationCollection                  ttip_c;
    reco::TrackTauImpactParameterAssociationMapType                     ttip_cm;
    reco::TrackTauImpactParameterAssociationRefType                     ttip_cr;
    reco::TauImpactParameterTrackData                                   tipd;
    reco::TauImpactParameterInfo                                        tip;
    reco::TauImpactParameterInfoCollection                              tip_c;
    reco::TauImpactParameterInfoRef                                     tip_r;
    reco::TauImpactParameterInfoRefProd                                 tip_rp;
    reco::TauImpactParameterInfoRefVector                               tip_rv;
    edm::Wrapper<reco::TauImpactParameterInfoCollection>                tip_wc;

    reco::TauMassTagInfo::ClusterTrackAssociation                       cta;
    reco::TauMassTagInfo::ClusterTrackAssociationCollection             cta_c;
    reco::TauMassTagInfo_ClusterTrackAssociationMapType                 cta_cm;
    reco::TauMassTagInfo_ClusterTrackAssociationRefType                 cta_cr;
    reco::TauMassTagInfo                                                tmt;
    reco::TauMassTagInfoCollection                                      tmt_c;
    reco::TauMassTagInfoRef                                             tmt_r;
    reco::TauMassTagInfoRefProd                                         tmt_rp;
    reco::TauMassTagInfoRefVector                                       tmt_rv;
    edm::Wrapper<reco::TauMassTagInfoCollection>                        tmt_wc;

    reco::TrackCountingTagInfo                                          tc;
    reco::TrackCountingTagInfoCollection                                tc_c;
    reco::TrackCountingTagInfoRef                                       tc_r;
    reco::TrackCountingTagInfoRefProd                                   tc_rp;
    reco::TrackCountingTagInfoRefVector                                 tc_rv;
    edm::Wrapper<reco::TrackCountingTagInfoCollection>                  tc_wc;

    reco::TrackProbabilityTagInfo                                       tp;
    reco::TrackProbabilityTagInfoCollection                             tp_c;
    reco::TrackProbabilityTagInfoRef                                    tp_r;
    reco::TrackProbabilityTagInfoRefProd                                tp_rp;
    reco::TrackProbabilityTagInfoRefVector                              tp_rv;
    edm::Wrapper<reco::TrackProbabilityTagInfoCollection>               tp_wc;


    reco::JetCrystalsAssociation                                        jca;
    reco::JetCrystalsAssociation::base_class                            jca_base;
    reco::JetCrystalsAssociationCollection                              jca_c;
    reco::JetCrystalsAssociationRef                                     jca_r;
    reco::JetCrystalsAssociationRefProd                                 jca_rp;
    reco::JetCrystalsAssociationRefVector                               jca_rv;
    edm::Wrapper<reco::JetCrystalsAssociationCollection>                jca_wc;

    reco::JetEisolAssociation                                           jea;
    reco::JetEisolAssociationCollection                                 jea_c;
    reco::JetEisolAssociationRef                                        jea_r;
    reco::JetEisolAssociationRefProd                                    jea_rp;
    reco::JetEisolAssociationRefVector                                  jea_rv;
    edm::Wrapper<reco::JetEisolAssociationCollection>                   jea_wc;



    reco::TrackIPTagInfo                                                tcip;
    reco::TrackIPTagInfo::TrackIPData                                   tcip_data;
    std::vector<reco::TrackIPTagInfo::TrackIPData>                      tcip_datav;
    reco::TrackIPTagInfoCollection                                      tcip_c;
    reco::TrackIPTagInfoRef                                             tcip_r;
    reco::TrackIPTagInfoRefProd                                         tcip_rp;
    reco::TrackIPTagInfoRefVector                                       tcip_rv;

    edm::Wrapper<reco::TrackIPTagInfoCollection>                        tcip_wc;

    reco::BaseTagInfo                                                   bti;
    reco::BaseTagInfoCollection                                         bti_c;
    reco::BaseTagInfoRef                                                bti_r;
    reco::BaseTagInfoRefProd                                            bti_rp;
    reco::BaseTagInfoRefVector                                          bti_rv;
    edm::Wrapper<reco::BaseTagInfoCollection>                           bti_wc;
    
    reco::JetTagInfo                                                    jti;
    reco::JetTagInfoCollection                                          jti_c;
    reco::JetTagInfoRef                                                 jti_r;
    reco::JetTagInfoRefProd                                             jti_rp;
    reco::JetTagInfoRefVector                                           jti_rv;
    edm::Wrapper<reco::JetTagInfoCollection>                            jti_wc;

    reco::JTATagInfo                                                    jtati;
    reco::JTATagInfoCollection                                          jtati_c;
    reco::JTATagInfoRef                                                 jtati_r;
    reco::JTATagInfoRefProd                                             jtati_rp;
    reco::JTATagInfoRefVector                                           jtati_rv;
    edm::Wrapper<reco::JTATagInfoCollection>                            jtati_wc;

    std::vector<Measurement1D>                                          vm1d;
	    
    // RefToBase Holders for TagInfos
    edm::RefToBase<reco::BaseTagInfo>                                           rb;
    edm::reftobase::IndirectHolder<reco::BaseTagInfo>                           rbh;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::BaseTagInfoRef>             rb_bti;
    edm::reftobase::RefHolder<reco::BaseTagInfoRef>                             rbh_bti;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::JTATagInfoRef>              rb_jtati;
    edm::reftobase::RefHolder<reco::JTATagInfoRef>                              rbh_jtati;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::JetTagInfoRef>              rb_jti;
    edm::reftobase::RefHolder<reco::JetTagInfoRef>                              rbh_jti;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::TrackCountingTagInfoRef>    rb_tc;
    edm::reftobase::RefHolder<reco::TrackCountingTagInfoRef>                    rbh_tc;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::TrackIPTagInfoRef>          rb_tcip;
    edm::reftobase::RefHolder<reco::TrackIPTagInfoRef>                          rbh_tcip;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::SecondaryVertexTagInfoRef>  rb_sv;
    edm::reftobase::RefHolder<reco::SecondaryVertexTagInfoRef>                  rbh_sv;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::CombinedTauTagInfoRef>      rb_ct;
    edm::reftobase::RefHolder<reco::CombinedTauTagInfoRef>                      rbh_ct;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::IsolatedTauTagInfoRef>      rb_it;
    edm::reftobase::RefHolder<reco::IsolatedTauTagInfoRef>                      rbh_it;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::SoftLeptonTagInfoRef>       rb_sl;
    edm::reftobase::RefHolder<reco::SoftLeptonTagInfoRef>                       rbh_sl;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::TauMassTagInfoRef>          rb_tmt;
    edm::reftobase::RefHolder<reco::TauMassTagInfoRef>                          rbh_tmt;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::TrackProbabilityTagInfoRef> rb_tp;
    edm::reftobase::RefHolder<reco::TrackProbabilityTagInfoRef>                 rbh_tp;
  }
}
