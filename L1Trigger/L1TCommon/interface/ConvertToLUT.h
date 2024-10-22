#ifndef L1Trigger_L1TCommon_l1t_ConvertToLUT
#define L1Trigger_L1TCommon_l1t_ConvertToLUT

#include <vector>
#include <sstream>
#include "CondFormats/L1TObjects/interface/LUT.h"

namespace l1t {

  inline l1t::LUT convertToLUT(const std::vector<uint64_t> &v, int padding = -1) noexcept {
    unsigned int addrWidth = (32 - __builtin_clz(uint32_t(v.size() - 1)));
    std::stringstream oss;
    oss << "#<header> V1 " << addrWidth << " 31 </header> " << std::endl;  // hardcode 32 bits for data
    for (unsigned int i = 0; i < v.size(); i++)
      oss << i << " " << v[i] << std::endl;
    // add padding to 2^addrWidth rows
    if (padding >= 0)
      for (unsigned int i = v.size(); i < (size_t)(1 << addrWidth); i++)
        oss << i << " " << padding << std::endl;

    l1t::LUT lut;
    lut.read(oss);

    return lut;
  }

  inline l1t::LUT convertToLUT(const std::vector<char> &v, int padding = -1) noexcept {
    return convertToLUT(std::vector<uint64_t>(v.begin(), v.end()), padding);
  }
  inline l1t::LUT convertToLUT(const std::vector<short> &v, int padding = -1) noexcept {
    return convertToLUT(std::vector<uint64_t>(v.begin(), v.end()), padding);
  }
  inline l1t::LUT convertToLUT(const std::vector<int> &v, int padding = -1) noexcept {
    return convertToLUT(std::vector<uint64_t>(v.begin(), v.end()), padding);
  }
  inline l1t::LUT convertToLUT(const std::vector<long> &v, int padding = -1) noexcept {
    return convertToLUT(std::vector<uint64_t>(v.begin(), v.end()), padding);
  }
  inline l1t::LUT convertToLUT(const std::vector<long long> &v, int padding = -1) noexcept {
    return convertToLUT(std::vector<uint64_t>(v.begin(), v.end()), padding);
  }
  inline l1t::LUT convertToLUT(const std::vector<unsigned char> &v, int padding = -1) noexcept {
    return convertToLUT(std::vector<uint64_t>(v.begin(), v.end()), padding);
  }
  inline l1t::LUT convertToLUT(const std::vector<unsigned short> &v, int padding = -1) noexcept {
    return convertToLUT(std::vector<uint64_t>(v.begin(), v.end()), padding);
  }
  inline l1t::LUT convertToLUT(const std::vector<unsigned int> &v, int padding = -1) noexcept {
    return convertToLUT(std::vector<uint64_t>(v.begin(), v.end()), padding);
  }
}  // namespace l1t

#endif
