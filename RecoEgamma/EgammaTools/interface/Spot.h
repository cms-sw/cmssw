#ifndef RecoEgamma_EgammaTools_Spot_h
#define RecoEgamma_EgammaTools_Spot_h

#include "DataFormats/DetId/interface/DetId.h"
#include <vector>

namespace hgcal {

  class Spot {
  public:
    Spot(DetId detid, double energy, const std::vector<double>& row, unsigned int layer, float fraction, double mip)
        : detId_(detid),
          energy_(energy),
          row_(row),
          layer_(layer),
          fraction_(fraction),
          mip_(mip),
          multiplicity_(int(energy / mip)),
          subdet_(detid.subdetId()),
          isCore_(fraction > 0.) {}
    ~Spot() {}
    inline DetId detId() const { return detId_; }
    inline float energy() const { return energy_; }
    inline const double* row() const { return &row_[0]; }
    inline float fraction() const { return fraction_; }
    inline float mip() const { return mip_; }
    inline int multiplicity() const { return multiplicity_; }
    inline unsigned int layer() const { return layer_; }
    inline int subdet() const { return subdet_; }
    inline bool isCore() const { return isCore_; }

  private:
    DetId detId_;
    float energy_;
    std::vector<double> row_;
    unsigned int layer_;
    float fraction_;
    float mip_;
    int multiplicity_;
    int subdet_;
    bool isCore_;
  };

}  // namespace hgcal

#endif
