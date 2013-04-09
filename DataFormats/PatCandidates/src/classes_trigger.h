#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/PtrVector.h"

#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "DataFormats/PatCandidates/interface/TriggerEvent.h"

namespace {
  struct dictionary {

  pat::TriggerObjectCollection v_p_to;
  pat::TriggerObjectCollection::const_iterator v_p_to_ci;
  edm::Wrapper<pat::TriggerObjectCollection> w_v_p_to;
  pat::TriggerObjectRef r_p_to;
  std::pair< std::string, pat::TriggerObjectRef > p_r_p_to;
  pat::TriggerObjectMatchMap m_r_p_to;
  pat::TriggerObjectRefProd rp_p_to;
  edm::Wrapper<pat::TriggerObjectRefProd> w_rp_p_to;
  pat::TriggerObjectRefVector rv_p_to;
  pat::TriggerObjectRefVectorIterator rv_p_to_i;
  pat::TriggerObjectMatch a_p_to;
  edm::reftobase::Holder<reco::Candidate, pat::TriggerObjectRef> h_p_to;
  edm::reftobase::RefHolder<pat::TriggerObjectRef> rh_p_to;
//   edm::reftobase::VectorHolder<reco::Candidate, pat::TriggerObjectRefVector> vh_p_to;
//   edm::reftobase::RefVectorHolder<pat::TriggerObjectRefVector> rvh_p_to;
  edm::Wrapper<pat::TriggerObjectMatch> w_a_p_to;
  pat::TriggerObjectMatchRefProd rp_a_p_to;
  std::pair< std::string, pat::TriggerObjectMatchRefProd > p_rp_a_p_to;
  pat::TriggerObjectMatchContainer m_rp_a_p_to;
  pat::TriggerObjectMatchContainer::const_iterator m_rp_a_p_to_ci;
  edm::Wrapper<pat::TriggerObjectMatchContainer> w_m_rp_a_p_to;

  pat::TriggerObjectStandAloneCollection v_p_tosa;
  pat::TriggerObjectStandAloneCollection::const_iterator v_p_tosa_ci;
  edm::Wrapper<pat::TriggerObjectStandAloneCollection> w_v_p_tosa;
  pat::TriggerObjectStandAloneRef r_p_tosa;
  pat::TriggerObjectStandAloneRefProd rp_p_tosa;
  edm::Wrapper<pat::TriggerObjectStandAloneRefProd> w_rp_p_tosa;
  pat::TriggerObjectStandAloneRefVector rv_p_tosa;
  pat::TriggerObjectStandAloneRefVectorIterator rv_p_tosa_i;
  pat::TriggerObjectStandAloneMatch a_p_tosa;
  edm::reftobase::Holder<reco::Candidate, pat::TriggerObjectStandAloneRef> h_p_tosa;
  edm::reftobase::RefHolder<pat::TriggerObjectStandAloneRef> rh_p_tosa;
//   edm::reftobase::VectorHolder<reco::Candidate, pat::TriggerObjectStandAloneRefVector> vh_p_tosa;
//   edm::reftobase::RefVectorHolder<pat::TriggerObjectStandAloneRefVector> rvh_p_tosa;
  edm::Wrapper<pat::TriggerObjectStandAloneMatch> w_a_p_tosa;

  pat::TriggerFilterCollection v_p_tf;
  pat::TriggerFilterCollection::const_iterator v_p_tf_ci;
  edm::Wrapper<pat::TriggerFilterCollection> w_v_p_tf;
  pat::TriggerFilterRef r_p_tf;
  pat::TriggerFilterRefProd rp_p_tf;
  edm::Wrapper<pat::TriggerFilterRefProd> w_rp_p_tf;
  pat::TriggerFilterRefVector rv_p_tf;
  pat::TriggerFilterRefVectorIterator rv_p_tf_i;

  pat::TriggerPathCollection v_p_tp;
  pat::TriggerPathCollection::const_iterator v_p_tp_ci;
  edm::Wrapper<pat::TriggerPathCollection> w_v_p_tp;
  pat::TriggerPathRef r_p_tp;
  pat::TriggerPathRefProd rp_p_tp;
  edm::Wrapper<pat::TriggerPathRefProd> w_rp_p_tp;
  pat::TriggerPathRefVector rv_p_tp;
  pat::TriggerPathRefVectorIterator rv_p_tp_i;
  pat::L1Seed p_bs;
  pat::L1SeedCollection vp_bs;
  pat::L1SeedCollection::const_iterator vp_bs_ci;

  pat::TriggerConditionCollection v_p_tc;
  pat::TriggerConditionCollection::const_iterator v_p_tc_ci;
  edm::Wrapper<pat::TriggerConditionCollection> w_v_p_tc;
  pat::TriggerConditionRef r_p_tc;
  pat::TriggerConditionRefProd rp_p_tc;
  edm::Wrapper<pat::TriggerConditionRefProd> w_rp_p_tc;
  pat::TriggerConditionRefVector rv_p_tc;
  pat::TriggerConditionRefVectorIterator rv_p_tc_i;

  pat::TriggerAlgorithmCollection v_p_ta;
  pat::TriggerAlgorithmCollection::const_iterator v_p_ta_ci;
  edm::Wrapper<pat::TriggerAlgorithmCollection> w_v_p_ta;
  pat::TriggerAlgorithmRef r_p_ta;
  pat::TriggerAlgorithmRefProd rp_p_ta;
  edm::Wrapper<pat::TriggerAlgorithmRefProd> w_rp_p_ta;
  pat::TriggerAlgorithmRefVector rv_p_ta;
  pat::TriggerAlgorithmRefVectorIterator rv_p_ta_i;

  edm::Wrapper<pat::TriggerEvent> w_p_te;

  };

}
