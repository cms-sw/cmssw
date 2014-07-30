#ifndef TrackReco_TrackResiduals_h
#define TrackReco_TrackResiduals_h

#include <iostream>
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackReco/interface/TrackResiduals.h"

class Trajectory;

namespace reco
{

class HitPattern;

class TrackResiduals
{

    friend class Trajectory;

public:

    enum ResidualType {
        X_Y_RESIDUALS,
        X_Y_PULLS
    };

    TrackResiduals();
    TrackResiduals(enum ResidualType);
    void setResidualXY(int idx, double residualX, double residualY);
    void setPullXY(int idx, double pullX, double pullY);
    void setResidualType(enum ResidualType);
    void print(std::ostream &stream = std::cout) const;
    void print(const HitPattern &, std::ostream &stream = std::cout) const;
    /// get the residual of the ith hit (needs the hit pattern to
    /// figure out which hits are valid)
    double residualX(int i, const HitPattern &) const;
    double residualY(int i, const HitPattern &) const;
    /// get the residual of the ith valid hit, with no regard
    /// for alignment with the HitPattern
    double residualX(int i) const;
    double residualY(int i) const;

protected:
    /// number of residuals stored
    enum { numResiduals = 0x40 };
    static double unpack_pull(unsigned char);
    static unsigned char pack_pull(double);
    static double unpack_residual(unsigned char);
    static unsigned char pack_residual(double);

private:
    /// residuals, bitpacked two hits to a char
    unsigned char residuals_[numResiduals];
    char residualType;
};

} // namespace reco

#endif

