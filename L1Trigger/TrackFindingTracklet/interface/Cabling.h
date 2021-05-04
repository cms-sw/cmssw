// This class holds a list of stubs that are in a given layer and DCT region
#ifndef L1Trigger_TrackFindingTracklet_interface_Cabling_h
#define L1Trigger_TrackFindingTracklet_interface_Cabling_h

#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/DTC.h"
#include "L1Trigger/TrackFindingTracklet/interface/DTCLink.h"

#include <vector>
#include <map>

namespace trklet {

  class Settings;

  class Cabling {
  public:
    Cabling(std::string dtcconfig, std::string moduleconfig, Settings const& settings);

    ~Cabling() = default;

    const std::string& dtc(int layer, int ladder, int module) const;

    void addphi(const std::string& dtc, double phi, int layer, int module);

    void writephirange() const;

    std::vector<std::string> DTCs() const;

  private:
    Settings const& settings_;
    std::vector<DTCLink> links_;
    std::map<std::string, DTC> dtcranges_;
    std::map<std::string, DTC> dtcs_;
    std::map<int, std::map<int, std::map<int, std::string> > > modules_;
  };

};  // namespace trklet
#endif
