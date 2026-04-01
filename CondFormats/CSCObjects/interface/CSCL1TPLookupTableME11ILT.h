#ifndef CondFormats_CSCObjects_CSCL1TPLookupTableME11ILT_h
#define CondFormats_CSCObjects_CSCL1TPLookupTableME11ILT_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include <vector>

#define DECLARE_CSCL1TP_LUT(NAME) \
  private: \
    std::vector<unsigned> NAME##_; \
  public: \
    void set_##NAME(t_lut lut); \
    unsigned NAME(unsigned idx) const; \
    unsigned NAME##_bit_width() const

#define DEFINE_CSCL1TP_LUT(CLASS, NAME) \
  void CLASS::set_##NAME(t_lut lut) { NAME##_ = std::move(lut); } \
  unsigned CLASS::NAME(unsigned idx) const { return NAME##_.at(idx); } \
  unsigned CLASS::NAME##_bit_width() const { return CSCL1TPLookupTableUtils::get_lut_bit_width(NAME##_); }

struct CSCL1TPLookupTableUtils {
  using t_lut = std::vector<unsigned>;

  static unsigned get_lut_bit_width(const t_lut& lut);
  static unsigned get_common_lut_bit_width(std::initializer_list<unsigned> luts_bit_width, std::string_view lut_group_name);
};

class CSCL1TPLookupTableME11ILT {
public:
  using t_lut = CSCL1TPLookupTableUtils::t_lut;

  DECLARE_CSCL1TP_LUT(GEM_pad_CSC_es_ME11b_even);
  DECLARE_CSCL1TP_LUT(GEM_pad_CSC_es_ME11a_even);
  DECLARE_CSCL1TP_LUT(GEM_pad_CSC_es_ME11b_odd);
  DECLARE_CSCL1TP_LUT(GEM_pad_CSC_es_ME11a_odd);

  DECLARE_CSCL1TP_LUT(GEM_roll_CSC_min_wg_ME11_even);
  DECLARE_CSCL1TP_LUT(GEM_roll_CSC_max_wg_ME11_even);
  DECLARE_CSCL1TP_LUT(GEM_roll_CSC_min_wg_ME11_odd);
  DECLARE_CSCL1TP_LUT(GEM_roll_CSC_max_wg_ME11_odd);

  DECLARE_CSCL1TP_LUT(CSC_slope_cosi_2to1_L1_ME11a_even);
  DECLARE_CSCL1TP_LUT(CSC_slope_cosi_2to1_L1_ME11a_odd);
  DECLARE_CSCL1TP_LUT(CSC_slope_cosi_3to1_L1_ME11a_even);
  DECLARE_CSCL1TP_LUT(CSC_slope_cosi_3to1_L1_ME11a_odd);

  DECLARE_CSCL1TP_LUT(CSC_slope_cosi_2to1_L1_ME11b_even);
  DECLARE_CSCL1TP_LUT(CSC_slope_cosi_2to1_L1_ME11b_odd);
  DECLARE_CSCL1TP_LUT(CSC_slope_cosi_3to1_L1_ME11b_even);
  DECLARE_CSCL1TP_LUT(CSC_slope_cosi_3to1_L1_ME11b_odd);

  DECLARE_CSCL1TP_LUT(CSC_slope_cosi_corr_L1_ME11a_even);
  DECLARE_CSCL1TP_LUT(CSC_slope_cosi_corr_L1_ME11b_even);
  DECLARE_CSCL1TP_LUT(CSC_slope_cosi_corr_L1_ME11a_odd);
  DECLARE_CSCL1TP_LUT(CSC_slope_cosi_corr_L1_ME11b_odd);

  DECLARE_CSCL1TP_LUT(CSC_slope_corr_L1_ME11a_even);
  DECLARE_CSCL1TP_LUT(CSC_slope_corr_L1_ME11b_even);
  DECLARE_CSCL1TP_LUT(CSC_slope_corr_L1_ME11a_odd);
  DECLARE_CSCL1TP_LUT(CSC_slope_corr_L1_ME11b_odd);
  DECLARE_CSCL1TP_LUT(CSC_slope_corr_L2_ME11a_even);
  DECLARE_CSCL1TP_LUT(CSC_slope_corr_L2_ME11b_even);
  DECLARE_CSCL1TP_LUT(CSC_slope_corr_L2_ME11a_odd);
  DECLARE_CSCL1TP_LUT(CSC_slope_corr_L2_ME11b_odd);

  DECLARE_CSCL1TP_LUT(es_diff_slope_L1_ME11a_even);
  DECLARE_CSCL1TP_LUT(es_diff_slope_L1_ME11a_odd);
  DECLARE_CSCL1TP_LUT(es_diff_slope_L1_ME11b_even);
  DECLARE_CSCL1TP_LUT(es_diff_slope_L1_ME11b_odd);
  DECLARE_CSCL1TP_LUT(es_diff_slope_L2_ME11a_even);
  DECLARE_CSCL1TP_LUT(es_diff_slope_L2_ME11a_odd);
  DECLARE_CSCL1TP_LUT(es_diff_slope_L2_ME11b_even);
  DECLARE_CSCL1TP_LUT(es_diff_slope_L2_ME11b_odd);

  unsigned es_diff_slope_bit_width() const;

  COND_SERIALIZABLE;
};

#endif
