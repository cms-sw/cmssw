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
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"
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
#include "DataFormats/BTauReco/interface/CandSecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/BoostedDoubleSVTagInfo.h"
#include "DataFormats/BTauReco/interface/BoostedDoubleSVTagInfoFeatures.h"
#include "DataFormats/BTauReco/interface/ShallowTagInfo.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/TauImpactParameterInfo.h"
#include "DataFormats/BTauReco/interface/TauMassTagInfo.h"
#include "DataFormats/BTauReco/interface/JetEisolAssociation.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "DataFormats/BTauReco/interface/JetTagInfo.h"
#include "DataFormats/BTauReco/interface/CATopJetTagInfo.h"
#include "DataFormats/BTauReco/interface/HTTTopJetTagInfo.h"
#include "DataFormats/BTauReco/interface/JetFeatures.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexFeatures.h"
#include "DataFormats/BTauReco/interface/ShallowTagInfoFeatures.h"
#include "DataFormats/BTauReco/interface/NeutralCandidateFeatures.h"
#include "DataFormats/BTauReco/interface/ChargedCandidateFeatures.h"
#include "DataFormats/BTauReco/interface/DeepFlavourFeatures.h"
#include "DataFormats/BTauReco/interface/DeepFlavourTagInfo.h"
#include "DataFormats/BTauReco/interface/DeepDoubleXFeatures.h"
#include "DataFormats/BTauReco/interface/DeepDoubleXTagInfo.h"
#include "DataFormats/BTauReco/interface/DeepBoostedJetTagInfo.h"



namespace reco {
    typedef TrackTauImpactParameterAssociationCollection::map_type          TrackTauImpactParameterAssociationMapType;
    typedef TrackTauImpactParameterAssociationCollection::ref_type          TrackTauImpactParameterAssociationRefType;
    typedef TauMassTagInfo::ClusterTrackAssociationCollection::map_type     TauMassTagInfo_ClusterTrackAssociationMapType;
    typedef TauMassTagInfo::ClusterTrackAssociationCollection::ref_type     TauMassTagInfo_ClusterTrackAssociationRefType;
}

namespace DataFormats_BTauReco {
  struct dictionary {

    reco::btag::TrackData                                               sv_td;
    reco::btag::IndexedTrackData                                        sv_itd;
    std::vector<reco::btag::IndexedTrackData>                           sv_itdv;

    reco::SecondaryVertexTagInfo::VertexData                            sv_vd;
    std::vector<reco::SecondaryVertexTagInfo::VertexData>               sv_vdv;
    reco::SecondaryVertexTagInfo                                        sv;
    reco::SecondaryVertexTagInfoCollection                              sv_c;
    reco::SecondaryVertexTagInfoRef                                     sv_r;
    reco::SecondaryVertexTagInfoFwdRef                                  sv_fr;
    reco::SecondaryVertexTagInfoRefProd                                 sv_rp;
    reco::SecondaryVertexTagInfoRefVector                               sv_rv;
    edm::Wrapper<reco::SecondaryVertexTagInfoCollection>                sv_wc;

    reco::CandSecondaryVertexTagInfo::VertexData                        casv_vd;
    std::vector<reco::CandSecondaryVertexTagInfo::VertexData>           casv_vdv;
    reco::CandSecondaryVertexTagInfo                                    casv;
    reco::CandSecondaryVertexTagInfoCollection                          casv_c;
    reco::CandSecondaryVertexTagInfoRef                                 casv_r;
    reco::CandSecondaryVertexTagInfoFwdRef                              casv_fr;
    reco::CandSecondaryVertexTagInfoRefProd                             casv_rp;
    reco::CandSecondaryVertexTagInfoRefVector                           casv_rv;
    edm::Wrapper<reco::CandSecondaryVertexTagInfoCollection>            casv_wc;

    reco::BoostedDoubleSVTagInfo                                        bdsv;
    reco::BoostedDoubleSVTagInfoCollection                              bdsv_c;
    reco::BoostedDoubleSVTagInfoRef                                     bdsv_r;
    reco::BoostedDoubleSVTagInfoFwdRef                                  bdsv_fr;
    reco::BoostedDoubleSVTagInfoRefProd                                 bdsv_rp;
    reco::BoostedDoubleSVTagInfoRefVector                               bdsv_rv;
    edm::Wrapper<reco::BoostedDoubleSVTagInfoCollection>                bdsv_wc;

    reco::ShallowTagInfo                                                 dnn;
    reco::ShallowTagInfoCollection                                       dnn_c;
    reco::ShallowTagInfoRef                                              dnn_r;
    reco::ShallowTagInfoFwdRef                                           dnn_fr;
    reco::ShallowTagInfoRefProd                                          dnn_rp;
    reco::ShallowTagInfoRefVector                                        dnn_rv;
    edm::Wrapper<reco::ShallowTagInfoCollection>                         dnn_wc;

    reco::CombinedTauTagInfo                                            ct;
    reco::CombinedTauTagInfoCollection                                  ct_c;
    reco::CombinedTauTagInfoRef                                         ct_r;
    reco::CombinedTauTagInfoFwdRef                                      ct_fr;
    reco::CombinedTauTagInfoRefProd                                     ct_rp;
    reco::CombinedTauTagInfoRefVector                                   ct_rv;
    edm::Wrapper<reco::CombinedTauTagInfoCollection>                    ct_wc;


    reco::EMIsolatedTauTagInfo                                          em;
    reco::EMIsolatedTauTagInfoCollection                                em_c;
    reco::EMIsolatedTauTagInfoRef                                       em_r;
    reco::EMIsolatedTauTagInfoFwdRef                                    em_fr;
    reco::EMIsolatedTauTagInfoRefProd                                   em_rp;
    reco::EMIsolatedTauTagInfoRefVector                                 em_rv;
    edm::Wrapper<reco::EMIsolatedTauTagInfoCollection>                  em_wc;

    reco::IsolatedTauTagInfo                                            it;
    reco::IsolatedTauTagInfoCollection                                  it_c;
    reco::IsolatedTauTagInfoRef                                         it_r;
    reco::IsolatedTauTagInfoFwdRef                                      it_fr;
    reco::IsolatedTauTagInfoRefProd                                     it_rp;
    reco::IsolatedTauTagInfoRefVector                                   it_rv;
    edm::Wrapper<reco::IsolatedTauTagInfoCollection>                    it_wc;


    reco::SoftLeptonProperties                                          slp;

    std::pair<edm::RefToBase<reco::Track>, reco::SoftLeptonProperties>  slp_p;
    reco::SoftLeptonTagInfo::LeptonMap                                  slp_m;
    reco::SoftLeptonTagInfo                                             sl;
    reco::SoftLeptonTagInfoCollection                                   sl_c;
    reco::SoftLeptonTagInfoRef                                          sl_r;
    reco::SoftLeptonTagInfoFwdRef                                       sl_fr;
    reco::SoftLeptonTagInfoRefProd                                      sl_rp;
    reco::SoftLeptonTagInfoRefVector                                    sl_rv;
    edm::Wrapper<reco::SoftLeptonTagInfoCollection>                     sl_wc;

    std::pair<edm::Ptr<reco::Candidate>, reco::SoftLeptonProperties>    cslp_p;
    reco::CandSoftLeptonTagInfo::LeptonMap                                  cslp_m;
    reco::CandSoftLeptonTagInfo                                             csl;
    reco::CandSoftLeptonTagInfoCollection                                   csl_c;
    reco::CandSoftLeptonTagInfoRef                                          csl_r;
    reco::CandSoftLeptonTagInfoFwdRef                                       csl_fr;
    reco::CandSoftLeptonTagInfoRefProd                                      csl_rp;
    reco::CandSoftLeptonTagInfoRefVector                                    csl_rv;
    edm::Wrapper<reco::CandSoftLeptonTagInfoCollection>                     csl_wc;

    std::pair< reco::btau::TaggingVariableName, float >                 ptt1;
    std::vector<std::pair<reco::btau::TaggingVariableName,float> >      vptt1;
    reco::TaggingVariableList                                           tvl;
    reco::TaggingVariableListCollection                                 tvl_c;
    reco::TaggingVariableListRef                                        tvl_r;
    reco::TaggingVariableListFwdRef                                     tvl_fr;
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
    reco::TauImpactParameterInfoFwdRef                                  tip_fr;
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
    reco::TauMassTagInfoFwdRef                                          tmt_fr;
    reco::TauMassTagInfoRefProd                                         tmt_rp;
    reco::TauMassTagInfoRefVector                                       tmt_rv;
    edm::Wrapper<reco::TauMassTagInfoCollection>                        tmt_wc;

    reco::TrackCountingTagInfo                                          tc;
    reco::TrackCountingTagInfoCollection                                tc_c;
    reco::TrackCountingTagInfoRef                                       tc_r;
    reco::TrackCountingTagInfoFwdRef                                    tc_fr;
    reco::TrackCountingTagInfoRefProd                                   tc_rp;
    reco::TrackCountingTagInfoRefVector                                 tc_rv;
    edm::Wrapper<reco::TrackCountingTagInfoCollection>                  tc_wc;

    reco::TrackProbabilityTagInfo                                       tp;
    reco::TrackProbabilityTagInfoCollection                             tp_c;
    reco::TrackProbabilityTagInfoRef                                    tp_r;
    reco::TrackProbabilityTagInfoFwdRef                                 tp_fr;
    reco::TrackProbabilityTagInfoRefProd                                tp_rp;
    reco::TrackProbabilityTagInfoRefVector                              tp_rv;
    edm::Wrapper<reco::TrackProbabilityTagInfoCollection>               tp_wc;


    reco::JetCrystalsAssociation                                        jca;
    reco::JetCrystalsAssociation::base_class                            jca_base;
    reco::JetCrystalsAssociationCollection                              jca_c;
    reco::JetCrystalsAssociationRef                                     jca_r;
    reco::JetCrystalsAssociationFwdRef                                  jca_fr;
    reco::JetCrystalsAssociationRefProd                                 jca_rp;
    reco::JetCrystalsAssociationRefVector                               jca_rv;
    edm::Wrapper<reco::JetCrystalsAssociationCollection>                jca_wc;

    reco::JetEisolAssociation                                           jea;
    reco::JetEisolAssociationCollection                                 jea_c;
    reco::JetEisolAssociationRef                                        jea_r;
    reco::JetEisolAssociationFwdRef                                     jea_fr;
    reco::JetEisolAssociationRefProd                                    jea_rp;
    reco::JetEisolAssociationRefVector                                  jea_rv;
    edm::Wrapper<reco::JetEisolAssociationCollection>                   jea_wc;


    reco::btag::TrackIPData                                             tip_data;
    std::vector<reco::btag::TrackIPData>                                tip_datav;

    reco::TrackIPTagInfo                                                tcip;
    reco::TrackIPTagInfoCollection                                      tcip_c;
    reco::TrackIPTagInfoRef                                             tcip_r;
    reco::TrackIPTagInfoFwdRef                                          tcip_fr;
    reco::TrackIPTagInfoRefProd                                         tcip_rp;
    reco::TrackIPTagInfoRefVector                                       tcip_rv;
    edm::Wrapper<reco::TrackIPTagInfoCollection>                        tcip_wc;

    reco::CandIPTagInfo                                                 cdip;
    reco::CandIPTagInfoCollection                                       cdip_c;
    reco::CandIPTagInfoRef                                              cdip_r;
    reco::CandIPTagInfoFwdRef                                           cdip_fr;
    reco::CandIPTagInfoRefProd                                          cdip_rp;
    reco::CandIPTagInfoRefVector                                        cdip_rv;
    edm::Wrapper<reco::CandIPTagInfoCollection>                         cdip_wc;


    reco::BaseTagInfo                                                   bti;
    reco::BaseTagInfoCollection                                         bti_c;
    reco::BaseTagInfoRef                                                bti_r;
    reco::BaseTagInfoFwdRef                                             bti_fr;
    reco::BaseTagInfoRefProd                                            bti_rp;
    reco::BaseTagInfoRefVector                                          bti_rv;
    edm::Wrapper<reco::BaseTagInfoCollection>                           bti_wc;

    reco::JetTagInfo                                                    jti;
    reco::JetTagInfoCollection                                          jti_c;
    reco::JetTagInfoRef                                                 jti_r;
    reco::JetTagInfoFwdRef                                              jti_fr;
    reco::JetTagInfoRefProd                                             jti_rp;
    reco::JetTagInfoRefVector                                           jti_rv;
    edm::Wrapper<reco::JetTagInfoCollection>                            jti_wc;

    reco::JTATagInfo                                                    jtati;
    reco::JTATagInfoCollection                                          jtati_c;
    reco::JTATagInfoRef                                                 jtati_r;
    reco::JTATagInfoFwdRef                                              jtati_fr;
    reco::JTATagInfoRefProd                                             jtati_rp;
    reco::JTATagInfoRefVector                                           jtati_rv;
    edm::Wrapper<reco::JTATagInfoCollection>                            jtati_wc;

    reco::CATopJetProperties                                            catopjetp;
    std::pair<edm::RefToBase<reco::Jet>, reco::CATopJetProperties>      catopjetp_p;
    reco::CATopJetTagInfo                                               catopjet;
    reco::CATopJetTagInfoCollection                                     catopjet_c;
    reco::CATopJetTagInfoRef                                            catopjet_r;
    reco::CATopJetTagInfoRefProd                                        catopjet_rp;
    reco::CATopJetTagInfoRefVector                                      catopjet_rv;
    edm::Wrapper<reco::CATopJetTagInfoCollection>                       catopjet_wc;

    reco::HTTTopJetProperties                                            htttopjetp;
    std::pair<edm::RefToBase<reco::Jet>, reco::HTTTopJetProperties>      htttopjetp_p;

    reco::HTTTopJetTagInfo                                               htttopjet;
    reco::HTTTopJetTagInfoCollection                                     htttopjet_c;
    reco::HTTTopJetTagInfoRef                                            htttopjet_r;
    reco::HTTTopJetTagInfoRefProd                                        htttopjet_rp;
    reco::HTTTopJetTagInfoRefVector                                      htttopjet_rv;
    edm::Wrapper<reco::HTTTopJetTagInfoCollection>                       htttopjet_wc;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::HTTTopJetTagInfoRef> rb_htttopjet;
    edm::reftobase::RefHolder<reco::HTTTopJetTagInfoRef>                 rbh_htttopjet;

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
    edm::reftobase::Holder<reco::BaseTagInfo, reco::CandIPTagInfoRef>           rb_cdip;
    edm::reftobase::RefHolder<reco::CandIPTagInfoRef>                           rbh_cdip;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::SecondaryVertexTagInfoRef>  rb_sv;
    edm::reftobase::RefHolder<reco::SecondaryVertexTagInfoRef>                  rbh_sv;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::CandSecondaryVertexTagInfoRef>  rb_casv;
    edm::reftobase::RefHolder<reco::CandSecondaryVertexTagInfoRef>                  rbh_casv;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::BoostedDoubleSVTagInfoRef>  rb_bdsv;
    edm::reftobase::RefHolder<reco::BoostedDoubleSVTagInfoRef>                  rbh_bdsv;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::ShallowTagInfoRef>           rb_dnn;
    edm::reftobase::RefHolder<reco::ShallowTagInfoRef>                           rbh_dnn;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::DeepFlavourTagInfoRef>       rb_df;
    edm::reftobase::RefHolder<reco::DeepFlavourTagInfoRef>                       rbh_df;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::CombinedTauTagInfoRef>      rb_ct;
    edm::reftobase::RefHolder<reco::CombinedTauTagInfoRef>                      rbh_ct;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::IsolatedTauTagInfoRef>      rb_it;
    edm::reftobase::RefHolder<reco::IsolatedTauTagInfoRef>                      rbh_it;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::SoftLeptonTagInfoRef>       rb_sl;
    edm::reftobase::RefHolder<reco::SoftLeptonTagInfoRef>                       rbh_sl;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::CandSoftLeptonTagInfoRef>   rb_csl;
    edm::reftobase::RefHolder<reco::CandSoftLeptonTagInfoRef>                   rbh_csl;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::TauMassTagInfoRef>          rb_tmt;
    edm::reftobase::RefHolder<reco::TauMassTagInfoRef>                          rbh_tmt;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::TrackProbabilityTagInfoRef> rb_tp;
    edm::reftobase::RefHolder<reco::TrackProbabilityTagInfoRef>                 rbh_tp;

    edm::reftobase::Holder<reco::BaseTagInfo, reco::BaseTagInfoFwdRef>             rb_btif;
    edm::reftobase::RefHolder<reco::BaseTagInfoFwdRef>                             rbh_btif;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::JTATagInfoFwdRef>              rb_jtatif;
    edm::reftobase::RefHolder<reco::JTATagInfoFwdRef>                              rbh_jtatif;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::JetTagInfoFwdRef>              rb_jtif;
    edm::reftobase::RefHolder<reco::JetTagInfoFwdRef>                              rbh_jtif;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::TrackCountingTagInfoFwdRef>    rb_tcf;
    edm::reftobase::RefHolder<reco::TrackCountingTagInfoFwdRef>                    rbh_tcf;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::TrackIPTagInfoFwdRef>          rb_tcipf;
    edm::reftobase::RefHolder<reco::TrackIPTagInfoFwdRef>                          rbh_tcipf;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::CandIPTagInfoFwdRef>           rb_cdipf;
    edm::reftobase::RefHolder<reco::CandIPTagInfoFwdRef>                           rbh_cdipf;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::SecondaryVertexTagInfoFwdRef>  rb_svf;
    edm::reftobase::RefHolder<reco::SecondaryVertexTagInfoFwdRef>                  rbh_svf;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::CandSecondaryVertexTagInfoFwdRef>  rb_casvf;
    edm::reftobase::RefHolder<reco::CandSecondaryVertexTagInfoFwdRef>                  rbh_casvf;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::BoostedDoubleSVTagInfoFwdRef>  rb_bdsvf;
    edm::reftobase::RefHolder<reco::BoostedDoubleSVTagInfoFwdRef>                  rbh_bdsvf;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::ShallowTagInfoFwdRef>          rb_dnnf;
    edm::reftobase::RefHolder<reco::ShallowTagInfoFwdRef>                          rbh_dnnf;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::DeepFlavourTagInfoFwdRef>      rb_dff;
    edm::reftobase::RefHolder<reco::DeepFlavourTagInfoFwdRef>                      rbh_dff;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::CombinedTauTagInfoFwdRef>      rb_ctf;
    edm::reftobase::RefHolder<reco::CombinedTauTagInfoFwdRef>                      rbh_ctf;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::IsolatedTauTagInfoFwdRef>      rb_itf;
    edm::reftobase::RefHolder<reco::IsolatedTauTagInfoFwdRef>                      rbh_itf;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::SoftLeptonTagInfoFwdRef>       rb_slf;
    edm::reftobase::RefHolder<reco::SoftLeptonTagInfoFwdRef>                       rbh_slf;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::CandSoftLeptonTagInfoFwdRef>   rb_cslf;
    edm::reftobase::RefHolder<reco::CandSoftLeptonTagInfoFwdRef>                   rbh_cslf;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::TauMassTagInfoFwdRef>          rb_tmtf;
    edm::reftobase::RefHolder<reco::TauMassTagInfoFwdRef>                          rbh_tmtf;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::TrackProbabilityTagInfoFwdRef> rb_tpf;
    edm::reftobase::RefHolder<reco::TrackProbabilityTagInfoFwdRef>                 rbh_tpf;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::CATopJetTagInfoRef>            rb_catopjet;
    edm::reftobase::RefHolder<reco::CATopJetTagInfoRef>                            rbh_catopjet;

    edm::ValueMap<edm::Ptr<reco::BaseTagInfo> > vm_ptr_bti;
    edm::Wrapper<edm::ValueMap<edm::Ptr<reco::BaseTagInfo> > > w_vm_ptr_bti;
    std::vector<reco::BaseTagInfo*> pv_bti;
    edm::OwnVector<reco::BaseTagInfo, edm::ClonePolicy<reco::BaseTagInfo> > ov_bti;
    edm::Wrapper<edm::OwnVector<reco::BaseTagInfo, edm::ClonePolicy<reco::BaseTagInfo> > > w_ov_bti;
    edm::Ptr<reco::BaseTagInfo> ptr_bti;
    std::vector<edm::Ptr<reco::BaseTagInfo> > v_ptr_bti;
    edm::FwdPtr<reco::BaseTagInfo> fptr_bti;
    std::vector<edm::FwdPtr<reco::BaseTagInfo> > v_fptr_bti;


    edm::Ref<edm::OwnVector<reco::BaseTagInfo, edm::ClonePolicy<reco::BaseTagInfo> > > r_ov_bti;
    edm::FwdRef<edm::OwnVector<reco::BaseTagInfo, edm::ClonePolicy<reco::BaseTagInfo> > > fr_ov_bti;
    std::vector<edm::Ref<edm::OwnVector<reco::BaseTagInfo, edm::ClonePolicy<reco::BaseTagInfo> > > > v_r_ov_bti;
    std::vector<edm::FwdRef<edm::OwnVector<reco::BaseTagInfo, edm::ClonePolicy<reco::BaseTagInfo> > > > v_fr_ov_bti;

    edm::Wrapper<edm::Ref<edm::OwnVector<reco::BaseTagInfo, edm::ClonePolicy<reco::BaseTagInfo> > > > w_r_ov_bti;
    edm::Wrapper<edm::FwdRef<edm::OwnVector<reco::BaseTagInfo, edm::ClonePolicy<reco::BaseTagInfo> > > > w_fr_ov_bti;
    edm::Wrapper<std::vector<edm::Ref<edm::OwnVector<reco::BaseTagInfo, edm::ClonePolicy<reco::BaseTagInfo> > > > > w_v_r_ov_bti;
    edm::Wrapper<std::vector<edm::FwdRef<edm::OwnVector<reco::BaseTagInfo, edm::ClonePolicy<reco::BaseTagInfo> > > > > w_v_fr_ov_bti;

    // Dictionaries for SVTagInfoProxy
    edm::helpers::KeyVal<edm::RefProd<std::vector<reco::SecondaryVertexTagInfo> >, edm::RefProd<std::vector<reco::Vertex> > >  dummy03;
    edm::AssociationMap<edm::OneToMany<reco::SecondaryVertexTagInfoCollection, reco::VertexCollection> > dummy05;
    edm::Wrapper<edm::helpers::KeyVal<edm::RefProd<std::vector<reco::SecondaryVertexTagInfo> >, edm::RefProd<std::vector<reco::Vertex> > > > dummy04;
    edm::Wrapper<edm::AssociationMap<edm::OneToMany<std::vector<reco::SecondaryVertexTagInfo>, std::vector<reco::Vertex>, unsigned int > > > dummy06;

    edm::helpers::KeyVal<edm::RefProd<std::vector<reco::CandSecondaryVertexTagInfo> >, edm::RefProd<std::vector<reco::VertexCompositePtrCandidate> > >  dummy07;
    edm::AssociationMap<edm::OneToMany<reco::CandSecondaryVertexTagInfoCollection, std::vector<reco::VertexCompositePtrCandidate> > > dummy08;
    edm::Wrapper<edm::helpers::KeyVal<edm::RefProd<std::vector<reco::CandSecondaryVertexTagInfo> >, edm::RefProd<std::vector<reco::VertexCompositePtrCandidate> > > > dummy09;
    edm::Wrapper<edm::AssociationMap<edm::OneToMany<std::vector<reco::CandSecondaryVertexTagInfo>, std::vector<reco::VertexCompositePtrCandidate>, unsigned int > > > dummy10;

    btagbtvdeep::JetFeatures jet_features;
    btagbtvdeep::SecondaryVertexFeatures secondary_vertex_features;
    btagbtvdeep::ShallowTagInfoFeatures shallow_tag_info_features;
    btagbtvdeep::NeutralCandidateFeatures neutral_candidate_features;
    btagbtvdeep::ChargedCandidateFeatures charged_candidate_features;
    btagbtvdeep::DeepFlavourFeatures deep_flavour_features;

    reco::DeepFlavourTagInfo deep_flavour_tag_info;
    reco::DeepFlavourTagInfoCollection deep_flavour_tag_info_collection;
    reco::DeepFlavourTagInfoRef deep_flavour_tag_info_collection_ref;
    reco::DeepFlavourTagInfoFwdRef deep_flavour_tag_info_collection_fwd_ref;
    reco::DeepFlavourTagInfoRefProd deep_flavour_tag_info_collection_ref_prod;
    reco::DeepFlavourTagInfoRefVector deep_flavour_tag_info_collection_ref_vector;
    edm::Wrapper<reco::DeepFlavourTagInfoCollection> deep_flavour_tag_info_collection_edm_wrapper;

    btagbtvdeep::BoostedDoubleSVTagInfoFeatures boosted_double_sv_tag_info_features;
    btagbtvdeep::DeepDoubleXFeatures deep_doublex_features;
    reco::DeepDoubleXTagInfo deep_doublex_tag_info;
    reco::DeepDoubleXTagInfoCollection deep_doublex_tag_info_collection;
    reco::DeepDoubleXTagInfoRef deep_doublex_tag_info_collection_ref;
    reco::DeepDoubleXTagInfoFwdRef deep_doublex_tag_info_collection_fwd_ref;
    reco::DeepDoubleXTagInfoRefProd deep_doublex_tag_info_collection_ref_prod;
    reco::DeepDoubleXTagInfoRefVector deep_doublex_tag_info_collection_ref_vector;
    edm::Wrapper<reco::DeepDoubleXTagInfoCollection> deep_doublex_tag_info_collection_edm_wrapper;
    
    btagbtvdeep::DeepBoostedJetFeatures deep_boosted_jet_tag_info_features;
    reco::DeepBoostedJetTagInfo deep_boosted_jet_tag_info;
    reco::DeepBoostedJetTagInfoCollection deep_boosted_jet_tag_info_collection;
    reco::DeepBoostedJetTagInfoRef deep_boosted_jet_tag_info_collection_ref;
    reco::DeepBoostedJetTagInfoFwdRef deep_boosted_jet_tag_info_collection_fwd_ref;
    reco::DeepBoostedJetTagInfoRefProd deep_boosted_jet_tag_info_collection_ref_prod;
    reco::DeepBoostedJetTagInfoRefVector deep_boosted_jet_tag_info_collection_ref_vector;
    edm::Wrapper<reco::DeepBoostedJetTagInfoCollection> deep_boosted_jet_tag_info_collection_edm_wrapper;
  };
}
