#ifndef CONDCORE_PCLCONFIGPLUGINS_SIPIXELALIPCLTHRESHOLDSPAYLOADINSPECTORHELPER_H
#define CONDCORE_PCLCONFIGPLUGINS_SIPIXELALIPCLTHRESHOLDSPAYLOADINSPECTORHELPER_H

namespace PCLThresholdsPI {
  enum types { DELTA, SIG, MAXMOVE, MAXERR, FRACTION_CUT, END_OF_TYPES };

  /************************************************/
  inline const std::string getStringFromCoordEnum(const AlignPCLThresholds::coordType& coord) {
    switch (coord) {
      case AlignPCLThresholds::X:
        return "X";
      case AlignPCLThresholds::Y:
        return "Y";
      case AlignPCLThresholds::Z:
        return "Z";
      case AlignPCLThresholds::theta_X:
        return "#theta_{X}";
      case AlignPCLThresholds::theta_Y:
        return "#theta_{Y}";
      case AlignPCLThresholds::theta_Z:
        return "#theta_{Z}";
      default:
        return "should never be here";
    }
  }

  /************************************************/
  inline const std::string getStringFromTypeEnum(const types& type) {
    switch (type) {
      case types::DELTA:
        return "#Delta";
      case types::SIG:
        return "#Delta/#sigma ";
      case types::MAXMOVE:
        return "max. move ";
      case types::MAXERR:
        return "max. err ";
      case types::FRACTION_CUT:
        return "fraction cut ";
      default:
        return "should never be here";
    }
  }

  /************************************************/
  inline std::string replaceAll(const std::string& str, const std::string& from, const std::string& to) {
    std::string out(str);

    if (from.empty())
      return out;
    size_t start_pos = 0;
    while ((start_pos = out.find(from, start_pos)) != std::string::npos) {
      out.replace(start_pos, from.length(), to);
      start_pos += to.length();  // In case 'to' contains 'from', like replacing 'x' with 'yx'
    }
    return out;
  }

}  // namespace PCLThresholdsPI

#endif
