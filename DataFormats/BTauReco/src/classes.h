#include <utility>
#include <vector>

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/JetCrystalsAssociation.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfo.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/PFIsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/EMIsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/CombinedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/PFCombinedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/CombinedSVTagInfo.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/TauImpactParameterInfo.h"
#include "DataFormats/BTauReco/interface/TauMassTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackTauImpactParameterAssociation.h"
#include "DataFormats/BTauReco/interface/JetEisolAssociation.h"
//#include "DataFormats/BTauReco/interface/TrackIPData.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "DataFormats/BTauReco/interface/JetTagInfo.h"

namespace reco {
    typedef TrackTauImpactParameterAssociationCollection::map_type          TrackTauImpactParameterAssociationMapType;
    typedef TrackTauImpactParameterAssociationCollection::ref_type          TrackTauImpactParameterAssociationRefType;
    typedef TauMassTagInfo::ClusterTrackAssociationCollection::map_type     TauMassTagInfo_ClusterTrackAssociationMapType;
    typedef TauMassTagInfo::ClusterTrackAssociationCollection::ref_type     TauMassTagInfo_ClusterTrackAssociationRefType;
//  typedef JetTracksIPDataAssociationCollection::map_type                  JetTracksIPDataAssociationMapType;
//  typedef JetTracksIPDataAssociationCollection::ref_type                  JetTracksIPDataAssociationRefType;
}

namespace {
  namespace {

    reco::JetTag                                                        jt;
    reco::JetTagCollection                                              jt_c;
    reco::JetTagRef                                                     jt_r;
    reco::JetTagRefProd                                                 jt_rp;
    reco::JetTagRefVector                                               jt_rv;
    edm::Wrapper<reco::JetTagCollection>                                jt_wc;

    reco::CombinedSVTagInfo                                             sv;
    reco::CombinedSVTagInfoCollection                                   sv_c;
    reco::CombinedSVTagInfoRef                                          sv_r;
    reco::CombinedSVTagInfoRefProd                                      sv_rp;
    reco::CombinedSVTagInfoRefVector                                    sv_rv;
    edm::Wrapper<reco::CombinedSVTagInfoCollection>                     sv_wc;

    reco::CombinedTauTagInfo                                            ct;
    reco::CombinedTauTagInfoCollection                                  ct_c;
    reco::CombinedTauTagInfoRef                                         ct_r;
    reco::CombinedTauTagInfoRefProd                                     ct_rp;
    reco::CombinedTauTagInfoRefVector                                   ct_rv;
    edm::Wrapper<reco::CombinedTauTagInfoCollection>                    ct_wc;

    reco::PFCombinedTauTagInfo                                          pfct;
    reco::PFCombinedTauTagInfoCollection                                pfct_c;
    reco::PFCombinedTauTagInfoRef                                       pfct_r;
    reco::PFCombinedTauTagInfoRefProd                                   pfct_rp;
    reco::PFCombinedTauTagInfoRefVector                                 pfct_rv;
    edm::Wrapper<reco::PFCombinedTauTagInfoCollection>                  pfct_wc;

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

    reco::PFIsolatedTauTagInfo                                          pfit;
    reco::PFIsolatedTauTagInfoCollection                                pfit_c;
    reco::PFIsolatedTauTagInfoRef                                       pfit_r;
    reco::PFIsolatedTauTagInfoRefProd                                   pfit_rp;
    reco::PFIsolatedTauTagInfoRefVector                                 pfit_rv;
    edm::Wrapper<reco::PFIsolatedTauTagInfoCollection>                  pfit_wc;

    reco::SoftLeptonProperties                                          slp;
    std::pair<reco::TrackRef, reco::SoftLeptonProperties>               slp_p;
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

    reco::JetTracksAssociation                                          jta;
    reco::JetTracksAssociationCollection                                jta_c;
    reco::JetTracksAssociationRef                                       jta_r;
    reco::JetTracksAssociationRefProd                                   jta_rp;
    reco::JetTracksAssociationRefVector                                 jta_rv;
    edm::Wrapper<reco::JetTracksAssociationCollection>                  jta_wc;

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

    /*
    reco::JetTracksIPDataAssociation                                    jtip;
    reco::JetTracksIPDataAssociationCollection                          jtip_c;
    reco::JetTracksIPDataAssociationMapType                             jtip_cm;
    reco::JetTracksIPDataAssociationRefType                             jtip_cr;
    reco::JetTracksIPDataAssociationRef                                 jtip_r;
    reco::JetTracksIPDataAssociationRefProd                             jtip_rp;
    reco::JetTracksIPDataAssociationRefVector                           jtip_rv;
    edm::Wrapper<reco::JetTracksIPDataAssociationCollection>            jtip_wc;
    */

    reco::TrackIPTagInfo                                                tcip;
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
    edm::reftobase::Holder<reco::BaseTagInfo, reco::BaseTagInfoRef>             rb_bti;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::JTATagInfoRef>              rb_jtati;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::JetTagInfoRef>              rb_jti;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::TrackCountingTagInfoRef>    rb_tc;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::TrackIPTagInfoRef>          rb_tcip;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::CombinedSVTagInfoRef>       rb_sv;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::CombinedTauTagInfoRef>      rb_ct;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::PFCombinedTauTagInfoRef>    rb_pfct;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::IsolatedTauTagInfoRef>      rb_it;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::PFIsolatedTauTagInfoRef>    rb_pfit;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::SoftLeptonTagInfoRef>       rb_sl;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::TauMassTagInfoRef>          rb_tmt;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::TrackProbabilityTagInfoRef> rb_tp;
  }
}
