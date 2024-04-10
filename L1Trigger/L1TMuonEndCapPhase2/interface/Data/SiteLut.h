
#ifndef L1Trigger_L1TMuonEndCapPhase2_SiteLut_h
#define L1Trigger_L1TMuonEndCapPhase2_SiteLut_h

#include <map>
#include <tuple>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace emtf::phase2::data {

  class SiteLut {
    // Static
  public:
    static const int kInvalid;

    // Member
  public:
    SiteLut();

    ~SiteLut();

    void update(const edm::Event&, const edm::EventSetup&);

    const int& lookup(const std::tuple<int, int, int>&) const;

  private:
    // Key: Subsystem, Station, Ring
    // Value: Site
    std::map<std::tuple<int, int, int>, int> lut_;
  };

}  // namespace emtf::phase2::data

#endif  // L1Trigger_L1TMuonEndCapPhase2_SiteLut_h
