#include "CondFormats/CSCObjects/interface/CSCL1TPLookupTableME11ILT.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <algorithm>

unsigned CSCL1TPLookupTableUtils::get_lut_bit_width(const std::vector<unsigned>& lut) {
  const auto iter = std::max_element(lut.begin(), lut.end());
  return iter == lut.end() || *iter == 0 ? 0 : std::bit_width(*iter);
}

unsigned CSCL1TPLookupTableUtils::get_common_lut_bit_width(std::initializer_list<unsigned> luts_bit_width,
                                                           std::string_view lut_group_name) {
  const auto min_max = std::minmax(luts_bit_width);
  if (min_max.first != min_max.second)
    throw cms::Exception("InvalidLookupTable") << "Inconsistent bit widths among " << lut_group_name << " LUTs: "
      << min_max.first << " vs " << min_max.second;
  return min_max.first;
}

#define DEFINE_LUT(NAME) DEFINE_CSCL1TP_LUT(CSCL1TPLookupTableME11ILT, NAME)

DEFINE_LUT(GEM_pad_CSC_es_ME11b_even);
DEFINE_LUT(GEM_pad_CSC_es_ME11a_even);
DEFINE_LUT(GEM_pad_CSC_es_ME11b_odd);
DEFINE_LUT(GEM_pad_CSC_es_ME11a_odd);

DEFINE_LUT(GEM_roll_CSC_min_wg_ME11_even);
DEFINE_LUT(GEM_roll_CSC_max_wg_ME11_even);
DEFINE_LUT(GEM_roll_CSC_min_wg_ME11_odd);
DEFINE_LUT(GEM_roll_CSC_max_wg_ME11_odd);

DEFINE_LUT(CSC_slope_cosi_2to1_L1_ME11a_even);
DEFINE_LUT(CSC_slope_cosi_2to1_L1_ME11a_odd);
DEFINE_LUT(CSC_slope_cosi_3to1_L1_ME11a_even);
DEFINE_LUT(CSC_slope_cosi_3to1_L1_ME11a_odd);

DEFINE_LUT(CSC_slope_cosi_2to1_L1_ME11b_even);
DEFINE_LUT(CSC_slope_cosi_2to1_L1_ME11b_odd);
DEFINE_LUT(CSC_slope_cosi_3to1_L1_ME11b_even);
DEFINE_LUT(CSC_slope_cosi_3to1_L1_ME11b_odd);

DEFINE_LUT(CSC_slope_cosi_corr_L1_ME11a_even);
DEFINE_LUT(CSC_slope_cosi_corr_L1_ME11b_even);
DEFINE_LUT(CSC_slope_cosi_corr_L1_ME11a_odd);
DEFINE_LUT(CSC_slope_cosi_corr_L1_ME11b_odd);

DEFINE_LUT(CSC_slope_corr_L1_ME11a_even);
DEFINE_LUT(CSC_slope_corr_L1_ME11b_even);
DEFINE_LUT(CSC_slope_corr_L1_ME11a_odd);
DEFINE_LUT(CSC_slope_corr_L1_ME11b_odd);
DEFINE_LUT(CSC_slope_corr_L2_ME11a_even);
DEFINE_LUT(CSC_slope_corr_L2_ME11b_even);
DEFINE_LUT(CSC_slope_corr_L2_ME11a_odd);
DEFINE_LUT(CSC_slope_corr_L2_ME11b_odd);

DEFINE_LUT(es_diff_slope_L1_ME11a_even);
DEFINE_LUT(es_diff_slope_L1_ME11a_odd);
DEFINE_LUT(es_diff_slope_L1_ME11b_even);
DEFINE_LUT(es_diff_slope_L1_ME11b_odd);
DEFINE_LUT(es_diff_slope_L2_ME11a_even);
DEFINE_LUT(es_diff_slope_L2_ME11a_odd);
DEFINE_LUT(es_diff_slope_L2_ME11b_even);
DEFINE_LUT(es_diff_slope_L2_ME11b_odd);

unsigned CSCL1TPLookupTableME11ILT::es_diff_slope_bit_width() const {
  return CSCL1TPLookupTableUtils::get_common_lut_bit_width({
      es_diff_slope_L1_ME11a_even_bit_width(),
      es_diff_slope_L1_ME11a_odd_bit_width(),
      es_diff_slope_L1_ME11b_even_bit_width(),
      es_diff_slope_L1_ME11b_odd_bit_width(),
      es_diff_slope_L2_ME11a_even_bit_width(),
      es_diff_slope_L2_ME11a_odd_bit_width(),
      es_diff_slope_L2_ME11b_even_bit_width(),
      es_diff_slope_L2_ME11b_odd_bit_width(),
  }, "es_diff_slope");
}

#undef DEFINE_LUT
