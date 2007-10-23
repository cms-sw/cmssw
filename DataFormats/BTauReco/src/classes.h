#include <utility>
#include <vector>

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/JetCrystalsAssociation.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfo.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/EMIsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/CombinedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/CombinedSVTagInfo.h"
#include "DataFormats/BTauReco/interface/CombinedSVTagInfoFwd.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/TauImpactParameterInfo.h"
#include "DataFormats/BTauReco/interface/TauMassTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackTauImpactParameterAssociation.h"
#include "DataFormats/BTauReco/interface/JetEisolAssociation.h"
#include "DataFormats/BTauReco/interface/TrackIPData.h"

namespace reco {
    typedef TrackTauImpactParameterAssociationCollection::map_type          TrackTauImpactParameterAssociationMapType;
    typedef TrackTauImpactParameterAssociationCollection::ref_type          TrackTauImpactParameterAssociationRefType;
    typedef TrackTauImpactParameterAssociationCollection::value_type        TrackTauImpactParameterAssociation;
    typedef TauMassTagInfo::ClusterTrackAssociationCollection::map_type     TauMassTagInfo_ClusterTrackAssociationMapType;
    typedef TauMassTagInfo::ClusterTrackAssociationCollection               TauMassTagInfo_ClusterTrackAssociationCollection;
    typedef TauMassTagInfo::ClusterTrackAssociationCollection::ref_type     TauMassTagInfo_ClusterTrackAssociationRefType;
    typedef TauMassTagInfo::ClusterTrackAssociationCollection::value_type   TauMassTagInfo_ClusterTrackAssociation;
    typedef JetTracksAssociationCollection::map_type                        JetTracksAssociationMapType;
    typedef JetTracksAssociationCollection::ref_type                        JetTracksAssociationRefType;
    typedef JetCrystalsAssociationCollection::map_type                      JetCrystalsAssociationMapType;
    typedef JetCrystalsAssociationCollection::ref_type                      JetCrystalsAssociationRefType;
    typedef JetEisolAssociationCollection::map_type                         JetEisolAssociationMapType;
    typedef JetEisolAssociationCollection::ref_type                         JetEisolAssociationRefType;
    typedef JetTracksIPDataAssociationCollection::map_type                  JetTracksIPDataAssociationMapType;
    typedef JetTracksIPDataAssociationCollection::ref_type                  JetTracksIPDataAssociationRefType;
}

namespace {
  namespace {

    reco::JetTag                                                        jt;
    reco::JetTagCollection                                              jt_c;
    reco::JetTagRef                                                     jt_r;
    reco::JetTagRefProd                                                 jt_rp;
    reco::JetTagRefVector                                               jt_rv;
    edm::Wrapper<reco::JetTagCollection>                                jt_wc;

    reco::BaseTagInfo                                                   bti;
    reco::BaseTagInfoCollection                                         bti_c;
    reco::BaseTagInfoRef                                                bti_r;
    reco::BaseTagInfoRefProd                                            bti_rp;
    reco::BaseTagInfoRefVector                                          bti_rv;
    edm::Wrapper<reco::BaseTagInfoCollection>                           bti_wc;

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
    std::pair<reco::TrackRef, reco::SoftLeptonProperties>               slp_p;
    reco::SoftLeptonTagInfo::LeptonMap                                  slp_m;

    reco::SoftLeptonTagInfo                                             sl;
    reco::SoftLeptonTagInfoCollection                                   sl_c;
    reco::SoftLeptonTagInfoRef                                          sl_r;
    reco::SoftLeptonTagInfoRefProd                                      sl_rp;
    reco::SoftLeptonTagInfoRefVector                                    sl_rv;
    edm::Wrapper<reco::SoftLeptonTagInfoCollection>                     sl_wc;

    reco::TaggingVariable                                               tv;
    std::vector<reco::TaggingVariable>                                  tv_v;
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

    reco::TauMassTagInfo_ClusterTrackAssociation                        cta;
    reco::TauMassTagInfo_ClusterTrackAssociationCollection              cta_c;
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
    reco::JetTracksAssociationMapType                                   jta_cm;
    reco::JetTracksAssociationRefType                                   jta_cr;
    reco::JetTracksAssociationRef                                       jta_r;
    reco::JetTracksAssociationRefProd                                   jta_rp;
    reco::JetTracksAssociationRefVector                                 jta_rv;
    edm::Wrapper<reco::JetTracksAssociationCollection>                  jta_wc;
 
    reco::JetCrystalsAssociation                                        jca;
    reco::JetCrystalsAssociationCollection                              jca_c;
    reco::JetCrystalsAssociationMapType                                 jca_cm;
    reco::JetCrystalsAssociationRefType                                 jca_cr;
    reco::JetCrystalsAssociationRef                                     jca_r;
    reco::JetCrystalsAssociationRefProd                                 jca_rp;
    reco::JetCrystalsAssociationRefVector                               jca_rv;
    edm::Wrapper<reco::JetCrystalsAssociationCollection>                jca_wc;

    reco::JetEisolAssociation                                           jea;
    reco::JetEisolAssociationCollection                                 jea_c;
    reco::JetEisolAssociationMapType                                    jea_cm;
    reco::JetEisolAssociationRefType                                    jea_cr;
    reco::JetEisolAssociationRef                                        jea_r;
    reco::JetEisolAssociationRefProd                                    jea_rp;
    reco::JetEisolAssociationRefVector                                  jea_rv;
    edm::Wrapper<reco::JetEisolAssociationCollection>                   jea_wc;

    reco::JetTracksIPDataAssociation                                    jtip;
    reco::JetTracksIPDataAssociationCollection                          jtip_c;
    reco::JetTracksIPDataAssociationMapType                             jtip_cm;
    reco::JetTracksIPDataAssociationRefType                             jtip_cr;
    reco::JetTracksIPDataAssociationRef                                 jtip_r;
    reco::JetTracksIPDataAssociationRefProd                             jtip_rp;
    reco::JetTracksIPDataAssociationRefVector                           jtip_rv;
    edm::Wrapper<reco::JetTracksIPDataAssociationCollection>            jtip_wc;

  }
}
