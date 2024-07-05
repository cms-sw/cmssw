#ifndef DataFormats_L1Scouting_L1ScoutingBMTFStub_h
#define DataFormats_L1Scouting_L1ScoutingBMTFStub_h

#include "DataFormats/L1Scouting/interface/OrbitCollection.h"

namespace l1ScoutingRun3 {

  class BMTFStub {
  public:
    BMTFStub()
        : hwPhi_(0), hwPhiB_(0), hwQual_(0), hwEta_(0), hwQEta_(0), station_(0), wheel_(0), sector_(0), tag_(0) {}

    BMTFStub(int hwPhi, int hwPhiB, int hwQual, int hwEta, int hwQEta, int station, int wheel, int sector, int tag)
        : hwPhi_(hwPhi),
          hwPhiB_(hwPhiB),
          hwQual_(hwQual),
          hwEta_(hwEta),
          hwQEta_(hwQEta),
          station_(station),
          wheel_(wheel),
          sector_(sector),
          tag_(tag) {}

    void setHwPhi(int hwPhi) { hwPhi_ = hwPhi; }
    void setHwPhiB(int hwPhiB) { hwPhiB_ = hwPhiB; }
    void setHwQual(int hwQual) { hwQual_ = hwQual; }
    void setHwEta(int hwEta) { hwEta_ = hwEta; }
    void setHwQEta(int hwQEta) { hwQEta_ = hwQEta; }
    void setStation(int station) { station_ = station; }
    void setWheel(int wheel) { wheel_ = wheel; }
    void setSector(int sector) { sector_ = sector; }
    void setTag(int tag) { tag_ = tag; }

    int hwPhi() const { return hwPhi_; }
    int hwPhiB() const { return hwPhiB_; }
    int hwQual() const { return hwQual_; }
    int hwEta() const { return hwEta_; }
    int hwQEta() const { return hwQEta_; }
    int station() const { return station_; }
    int wheel() const { return wheel_; }
    int sector() const { return sector_; }
    int tag() const { return tag_; }

  private:
    int hwPhi_;
    int hwPhiB_;
    int hwQual_;
    int hwEta_;
    int hwQEta_;
    int station_;
    int wheel_;
    int sector_;
    int tag_;
  };

  typedef OrbitCollection<BMTFStub> BMTFStubOrbitCollection;

}  // namespace l1ScoutingRun3

#endif  //DataFormats_L1Scouting_L1ScoutingBMTFStub_h
