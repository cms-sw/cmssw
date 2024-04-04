#ifndef L1Trigger_L1TMuonEndCapPhase2_ActivationLut_h
#define L1Trigger_L1TMuonEndCapPhase2_ActivationLut_h

#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"

namespace emtf::phase2::data {

  class ActivationLut {
  public:
    ActivationLut();

    ~ActivationLut();

    void update(const edm::Event&, const edm::EventSetup&);

    const trk_pt_t& lookupPromptPt(const trk_nn_address_t&) const;
    const trk_pt_t& lookupDispPt(const trk_nn_address_t&) const;
    const trk_rels_t& lookupRels(const trk_nn_address_t&) const;
    const trk_dxy_t& lookupDxy(const trk_nn_address_t&) const;

  private:
    std::vector<trk_pt_t> prompt_pt_lut_;
    std::vector<trk_pt_t> disp_pt_lut_;
    std::vector<trk_rels_t> rels_lut_;
    std::vector<trk_dxy_t> dxy_lut_;
  };

}  // namespace emtf::phase2::data

#endif  // L1Trigger_L1TMuonEndCapPhase2_ActivationLut_h
