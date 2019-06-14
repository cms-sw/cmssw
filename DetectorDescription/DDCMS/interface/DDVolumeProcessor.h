#ifndef DETECTOR_DESCRIPTION_DD_VOLUME_PROCESSOR_H
#define DETECTOR_DESCRIPTION_DD_VOLUME_PROCESSOR_H

#include "DD4hep/VolumeProcessor.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"

#include <string>
#include <string_view>
#include <vector>

namespace cms {

  class DDVolumeProcessor : public dd4hep::PlacedVolumeProcessor {
  public:
    using Volume = dd4hep::Volume;
    using PlacedVolume = dd4hep::PlacedVolume;
    using PlacedVolumeProcessor = dd4hep::PlacedVolumeProcessor;

    DDVolumeProcessor() = default;
    ~DDVolumeProcessor() override = default;

    std::string_view stripNamespace(std::string_view v) {
      auto first = v.find_first_of(":");
      v.remove_prefix(std::min(first + 1, v.size()));
      return v;
    }

    std::string_view stripCopyNo(std::string_view v) {
      auto found = v.find_last_of("_");
      if (found != v.npos) {
        v.remove_suffix(v.size() - found);
      }
      return v;
    }

    std::vector<std::string_view> split(std::string_view str, const char* delims) {
      std::vector<std::string_view> ret;

      std::string_view::size_type start = 0;
      auto pos = str.find_first_of(delims, start);
      while (pos != std::string_view::npos) {
        if (pos != start) {
          ret.emplace_back(str.substr(start, pos - start));
        }
        start = pos + 1;
        pos = str.find_first_of(delims, start);
      }
      if (start < str.length())
        ret.emplace_back(str.substr(start, str.length() - start));
      return ret;
    }

    bool compare(std::string_view s1, std::string_view s2) {
      if (s1 == s2)
        return true;
      edm::LogVerbatim("Geometry") << '\"' << s1 << "\" does not match \"" << s2 << "\"\n";
      return false;
    }

    /// Callback to retrieve PlacedVolume information of an entire Placement
    int process(PlacedVolume pv, int level, bool recursive) override {
      m_volumes.emplace_back(pv.name());
      int ret = PlacedVolumeProcessor::process(pv, level, recursive);
      m_volumes.pop_back();
      return ret;
    }

    /// Volume callback
    int operator()(PlacedVolume pv, int level) override {
      Volume vol = pv.volume();
      edm::LogVerbatim("Geometry").log([&level, &vol, this](auto& log) {
        log << "\nHierarchical level:" << level << "   Placement:";
        for (const auto& i : m_volumes)
          log << "/" << i << ", \n";
        log << "\n\tMaterial:" << vol.material().name() << "\tSolid:   " << vol.solid().name() << "\n";
      });
      ++m_count;
      return 1;
    }
    int count() const { return m_count; }

  private:
    int m_count = 0;
    std::vector<std::string_view> m_volumes;
  };
}  // namespace cms

#endif
