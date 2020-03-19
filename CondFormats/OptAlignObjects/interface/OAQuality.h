#ifndef OAQuality_H
#define OAQuality_H

enum OAQuality { oa_fixed, oa_calibrated, oa_unknown };

struct OAQualityTranslator {
  constexpr static const char* const name(OAQuality oaq) {
    constexpr const char* const c[] = {"fixed", "calibrated", "unknown"};
    return c[oaq];
  }

  static constexpr const OAQuality index(int ind) {
    switch (ind) {
      case 0:
        return oa_fixed;
        break;
      case 1:
        return oa_calibrated;
        break;
      case 2:
        return oa_unknown;
        break;
      default:
        return oa_unknown;
        break;
    }
  }
};
#endif
