#include "CalibTracker/SiPixelQuality/interface/SiPixelDetectorStatus.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"

namespace DataFormats_SiPixelStatus {
  struct dictionary {
    SiPixelRocStatus rs;
    std::vector<SiPixelRocStatus> v_rs;
    SiPixelModuleStatus ms;
    PixelFEDChannel pixelFEDChannel;
    std::map<int, PixelFEDChannel> m_pixFEDCh;
    std::pair<int, SiPixelModuleStatus> p_ms;
    std::map<int, SiPixelModuleStatus> m_ms;
    SiPixelDetectorStatus ss;
    edm::Wrapper<SiPixelDetectorStatus> w_ss;
  };
}  // namespace DataFormats_SiPixelStatus
