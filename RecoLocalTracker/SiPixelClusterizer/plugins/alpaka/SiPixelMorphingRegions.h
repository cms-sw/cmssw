#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_SiPixelMorphingRegions_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_SiPixelMorphingRegions_h

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <boost/algorithm/string.hpp>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Utilities/interface/Exception.h"

// Selection of the modules where the digi morphing is applied, shared by the Phase-1 and Phase-2 producers
namespace siPixelMorphing {

  typedef std::pair<uint32_t, uint32_t> range;
  typedef std::vector<range> region;

  inline std::vector<region> parseRegions(const std::vector<std::string>& regionStrings, size_t size) {
    std::vector<region> regions;
    for (auto const& str : regionStrings) {
      region reg;
      std::vector<std::string> ranges;
      boost::split(ranges, str, boost::is_any_of(","));
      if (ranges.size() != size) {
        throw cms::Exception("Configuration") << "[SiPixelDigiMorphing]:"
                                              << " invalid number of coordinates provided in " << str << " (" << size
                                              << " expected, " << ranges.size() << " provided)\n";
      }
      for (auto const& r : ranges) {
        std::vector<std::string> limits;
        boost::split(limits, r, boost::is_any_of("-"));
        try {
          if (limits.size() == 2) {
            reg.push_back(std::make_pair(std::stoi(limits.at(0)), std::stoi(limits.at(1))));
          } else if (limits.size() == 1) {
            reg.push_back(std::make_pair(std::stoi(limits.at(0)), std::stoi(limits.at(0))));
          } else {
            throw cms::Exception("Configuration") << "[SiPixelDigiMorphing]:"
                                                  << " invalid range format in '" << r << "' (expected 'A' or 'A-B')\n";
          }
        } catch (cms::Exception&) {
          throw;
        } catch (...) {
          throw cms::Exception("Configuration") << "[SiPixelDigiMorphing]:"
                                                << " invalid coordinate value provided in " << str << "\n";
        }
      }
      regions.push_back(reg);
    }
    return regions;
  }

  inline bool skipDetId(const TrackerTopology* tTopo,
                        const DetId& detId,
                        const std::vector<region>& theBarrelRegions,
                        const std::vector<region>& theEndcapRegions) {
    if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
      if (theBarrelRegions.empty()) {
        return true;
      } else {
        uint32_t layer = tTopo->pxbLayer(detId.rawId());
        uint32_t ladder = tTopo->pxbLadder(detId.rawId());
        uint32_t module = tTopo->pxbModule(detId.rawId());
        bool inRegion = false;
        for (auto const& reg : theBarrelRegions) {
          if ((layer >= reg.at(0).first && layer <= reg.at(0).second) &&
              (ladder >= reg.at(1).first && ladder <= reg.at(1).second) &&
              (module >= reg.at(2).first && module <= reg.at(2).second)) {
            inRegion = true;
            break;
          }
        }
        return !inRegion;
      }
    } else {
      if (theEndcapRegions.empty()) {
        return true;
      } else {
        uint32_t disk = tTopo->pxfDisk(detId.rawId());
        uint32_t blade = tTopo->pxfBlade(detId.rawId());
        uint32_t side = tTopo->pxfSide(detId.rawId());
        uint32_t panel = tTopo->pxfPanel(detId.rawId());
        bool inRegion = false;
        for (auto const& reg : theEndcapRegions) {
          if ((disk >= reg.at(0).first && disk <= reg.at(0).second) &&
              (blade >= reg.at(1).first && blade <= reg.at(1).second) &&
              (side >= reg.at(2).first && side <= reg.at(2).second) &&
              (panel >= reg.at(3).first && panel <= reg.at(3).second)) {
            inRegion = true;
            break;
          }
        }
        return !inRegion;
      }
    }
  }

}  // namespace siPixelMorphing

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_SiPixelMorphingRegions_h
