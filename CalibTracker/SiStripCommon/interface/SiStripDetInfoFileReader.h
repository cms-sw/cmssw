#ifndef CalibTracker_SiStripChannelGain_SiStripDetInfoFileReader_h
#define CalibTracker_SiStripChannelGain_SiStripDetInfoFileReader_h

#include <string>
#include "CalibFormats/SiStripObjects/interface/SiStripDetInfo.h"

namespace SiStripDetInfoFileReader {
  using DetInfo = SiStripDetInfo::DetInfo;

  constexpr static char const* const kDefaultFile = "CalibTracker/SiStripCommon/data/SiStripDetInfo.dat";

  /**
   * Read SiStrip detector info from a file
   */
  SiStripDetInfo read(std::string filePath);
};  // namespace SiStripDetInfoFileReader

#endif
