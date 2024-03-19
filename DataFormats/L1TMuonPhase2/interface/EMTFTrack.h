#ifndef DataFormats_L1TMuonPhase2_EMTFTrack_h
#define DataFormats_L1TMuonPhase2_EMTFTrack_h

#include <array>
#include <cstdint>
#include <vector>

namespace l1t::phase2 {

  class EMTFTrack {
  public:
    typedef std::vector<int16_t> features_t;
    typedef std::vector<uint16_t> site_hits_t;
    typedef std::vector<uint16_t> site_segs_t;
    typedef std::vector<uint8_t> site_mask_t;

    EMTFTrack();
    ~EMTFTrack() = default;

    // Setters
    void setEndcap(int16_t aEndcap) { endcap_ = aEndcap; }
    void setSector(int16_t aSector) { sector_ = aSector; }
    void setBx(int16_t aBx) { bx_ = aBx; }
    void setUnconstrained(bool aUnconstrained) { unconstrained_ = aUnconstrained; }
    void setValid(bool aValid) { valid_ = aValid; }

    void setModelPtAddress(int16_t aAddress) { model_pt_address_ = aAddress; }
    void setModelRelsAddress(int16_t aAddress) { model_rels_address_ = aAddress; }
    void setModelDxyAddress(int16_t aAddress) { model_dxy_address_ = aAddress; }
    void setModelPattern(int16_t aModelPattern) { model_pattern_ = aModelPattern; }
    void setModelQual(int16_t aModelQual) { model_qual_ = aModelQual; }
    void setModelPhi(int32_t aModelPhi) { model_phi_ = aModelPhi; }
    void setModelEta(int32_t aModelEta) { model_eta_ = aModelEta; }
    void setModelFeatures(const features_t& aModelFeatures) { model_features_ = aModelFeatures; }

    void setEmtfQ(int16_t aEmtfQ) { emtf_q_ = aEmtfQ; }
    void setEmtfPt(int32_t aEmtfPt) { emtf_pt_ = aEmtfPt; }
    void setEmtfRels(int32_t aEmtfRels) { emtf_rels_ = aEmtfRels; }
    void setEmtfD0(int32_t aEmtfD0) { emtf_d0_ = aEmtfD0; }
    void setEmtfZ0(int32_t aEmtfZ0) { emtf_z0_ = aEmtfZ0; }
    void setEmtfBeta(int32_t aEmtfBeta) { emtf_beta_ = aEmtfBeta; }
    void setEmtfModeV1(int16_t aEmtfModeV1) { emtf_mode_v1_ = aEmtfModeV1; }
    void setEmtfModeV2(int16_t aEmtfModeV2) { emtf_mode_v2_ = aEmtfModeV2; }

    void setSiteHits(const site_hits_t& aSiteHits) { site_hits_ = aSiteHits; }
    void setSiteSegs(const site_segs_t& aSiteSegs) { site_segs_ = aSiteSegs; }
    void setSiteMask(const site_mask_t& aSiteMask) { site_mask_ = aSiteMask; }
    void setSiteRMMask(const site_mask_t& aSiteMask) { site_rm_mask_ = aSiteMask; }

    // Getters
    int16_t endcap() const { return endcap_; }
    int16_t sector() const { return sector_; }
    int16_t bx() const { return bx_; }
    bool unconstrained() const { return unconstrained_; }
    bool valid() const { return valid_; }

    int16_t modelPtAddress() const { return model_pt_address_; }
    int16_t modelRelsAddress() const { return model_rels_address_; }
    int16_t modelDxyAddress() const { return model_dxy_address_; }
    int16_t modelPattern() const { return model_pattern_; }
    int16_t modelQual() const { return model_qual_; }
    int32_t modelPhi() const { return model_phi_; }
    int32_t modelEta() const { return model_eta_; }
    const features_t& modelFeatures() const { return model_features_; }

    int16_t emtfQ() const { return emtf_q_; }
    int32_t emtfPt() const { return emtf_pt_; }
    int32_t emtfRels() const { return emtf_rels_; }
    int32_t emtfD0() const { return emtf_d0_; }
    int32_t emtfZ0() const { return emtf_z0_; }
    int32_t emtfBeta() const { return emtf_beta_; }
    int16_t emtfModeV1() const { return emtf_mode_v1_; }
    int16_t emtfModeV2() const { return emtf_mode_v2_; }

    const site_hits_t& siteHits() const { return site_hits_; }
    const site_segs_t& siteSegs() const { return site_segs_; }
    const site_mask_t& siteMask() const { return site_mask_; }
    const site_mask_t& siteRMMask() const { return site_rm_mask_; }

  private:
    int16_t endcap_;
    int16_t sector_;
    int16_t bx_;
    bool unconstrained_;
    bool valid_;

    int16_t model_pt_address_;
    int16_t model_rels_address_;
    int16_t model_dxy_address_;
    int16_t model_pattern_;
    int16_t model_qual_;
    int32_t model_phi_;
    int32_t model_eta_;
    features_t model_features_;

    int16_t emtf_q_;
    int32_t emtf_pt_;
    int32_t emtf_rels_;
    int32_t emtf_d0_;
    int32_t emtf_z0_;
    int32_t emtf_beta_;
    int16_t emtf_mode_v1_;
    int16_t emtf_mode_v2_;

    site_hits_t site_hits_;
    site_segs_t site_segs_;
    site_mask_t site_mask_;
    site_mask_t site_rm_mask_;
  };

  typedef std::vector<EMTFTrack> EMTFTrackCollection;

}  // namespace l1t::phase2

#endif  // DataFormats_L1TMuonPhase2_EMTFTrack_h not defined
