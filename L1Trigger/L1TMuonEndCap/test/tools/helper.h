#include <string>
#include <sstream>
#include <cmath>

namespace {

  template <typename T>
  inline T deltaPhiInRadians(T phi1, T phi2) {
    T result = phi1 - phi2;  // same convention as reco::deltaPhi()
    constexpr T _twopi = M_PI*2.;
    result /= _twopi;
    result -= std::round(result);
    result *= _twopi;  // result in [-pi,pi]
    return result;
  }

  template <typename T>
  inline T deltaPhiInDegrees(T phi1, T phi2) {
    T result = phi1 - phi2;  // same convention as reco::deltaPhi()
    constexpr T _twopi = 360.;
    result /= _twopi;
    result -= std::round(result);
    result *= _twopi;  // result in [-180,180]
    return result;
  }

  template<typename T>
  T get_subword(T word, unsigned int msb, unsigned int lsb) {
    int len = msb-lsb+1;
    len = (len < 0) ? 0 : len;
    return (word >> (lsb)) & ((1u<<len)-1);
  }

  template<typename T>
  std::string print_subaddresses(T address) {
    int mode_inv = (address >> (30-4)) & ((1<<4)-1);
    std::stringstream ss;

    switch (mode_inv) {
    case 0:
    case 1:
    case 2:
    case 4:
    case 8:
      // Invalid addresses
      ss  << get_subword(address,29,26) << ":"
          << get_subword(address,25, 0);
      break;

    case 3:
    case 5:
    case 9:
    case 6:
    case 10:
    case 12:
      // 2-station addresses
      ss  << get_subword(address,29,26) << ":"
          << get_subword(address,25,21) << ":"
          << get_subword(address,20,20) << ":"
          << get_subword(address,19,19) << ":"
          << get_subword(address,18,16) << ":"
          << get_subword(address,15,13) << ":"
          << get_subword(address,12,10) << ":"
          << get_subword(address, 9, 9) << ":"
          << get_subword(address, 8, 0);
      break;

    case 7:
    case 11:
    case 13:
      // 3-station addresses (except 2-3-4)
      ss  << get_subword(address,29,26) << ":"
          << get_subword(address,25,21) << ":"
          << get_subword(address,20,20) << ":"
          << get_subword(address,19,17) << ":"
          << get_subword(address,16,14) << ":"
          << get_subword(address,13,13) << ":"
          << get_subword(address,12,12) << ":"
          << get_subword(address,11, 7) << ":"
          << get_subword(address, 6, 0);
      break;

    case 14:
      // 3-station addresses (only 2-3-4)
      ss  << get_subword(address,29,26) << ":"
          << get_subword(address,25,21) << ":"
          << get_subword(address,20,18) << ":"
          << get_subword(address,17,15) << ":"
          << get_subword(address,14,14) << ":"
          << get_subword(address,13,13) << ":"
          << get_subword(address,12, 7) << ":"
          << get_subword(address, 6, 0);
      break;

    case 15:
      // 4-station addresses
      ss  << get_subword(address,29,26) << ":"
          << get_subword(address,25,21) << ":"
          << get_subword(address,20,20) << ":"
          << get_subword(address,19,19) << ":"
          << get_subword(address,18,18) << ":"
          << get_subword(address,17,12) << ":"
          << get_subword(address,11, 7) << ":"
          << get_subword(address, 6, 0);
      break;

    default:
      break;
    }
    return ss.str();
  }

}  // namespace
