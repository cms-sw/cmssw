#ifndef L1Trigger_L1TMuonEndCapPhase2_ZoneLut_h
#define L1Trigger_L1TMuonEndCapPhase2_ZoneLut_h

#include <map>
#include <tuple>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace emtf::phase2::data {

  // Forward declarations
  class Zone;

  // Classes
  class ZoneLut {
  public:
    ZoneLut();

    ~ZoneLut();

    void update(const edm::Event&, const edm::EventSetup&);

    int getZones(const int&, const int&) const;

    int getZones(const int&, const int&, const int&) const;

  private:
    std::vector<Zone> zones_;
  };

  class Zone {
    friend ZoneLut;

  public:
    Zone() = default;

    ~Zone() = default;

    bool contains(const int&, const int&) const;

    bool contains(const int&, const int&, const int&) const;

  private:
    // Key: Host
    // Value: Theta Range
    std::map<int, std::pair<int, int>> lut_;
  };

}  // namespace emtf::phase2::data

#endif  // L1Trigger_L1TMuonEndCapPhase2_ZoneLut_h
