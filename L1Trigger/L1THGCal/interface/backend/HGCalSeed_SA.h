#ifndef L1Trigger_L1THGCal_HGCalSeed_SA_h
#define L1Trigger_L1THGCal_HGCalSeed_SA_h

namespace l1thgcfirmware {

  class HGCalSeed {
  public:
    HGCalSeed(float x, float y, float z, float energy) : x_(x), y_(y), z_(z), energy_(energy) {}

    ~HGCalSeed(){};

    float x() const { return x_; }
    float y() const { return y_; }
    float z() const { return z_; }
    float energy() const { return energy_; }

  private:
    float x_;
    float y_;
    float z_;
    float energy_;
  };

  typedef std::vector<HGCalSeed> HGCalSeedSACollection;

}  // namespace l1thgcfirmware

#endif
