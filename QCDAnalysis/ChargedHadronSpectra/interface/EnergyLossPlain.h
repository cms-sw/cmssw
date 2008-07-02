#ifndef _EnergyLossPlain_h_
#define _EnergyLossPlain_h_

#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

class SiPixelRecHit;
class SiStripRecHit2D;

class EnergyLossPlain
{
  public:
    EnergyLossPlain(const TrackerGeometry* theTracker_,
                    double pixelToStripMultiplier,
                    double pixelToStripExponent);
    ~EnergyLossPlain();
    void loadOptimalWeights();
    int estimate(const Trajectory* trajectory,
                 std::vector<std::pair<int,double> >& arithmeticMean,
                 std::vector<std::pair<int,double> >&  truncatedMean);

  private:
    double average (std::vector<double>& values);
    double truncate(std::vector<double>& values);
    double optimal (std::vector<double>& values);

    double logTruncate(std::vector<double>& values);
    double expected(double Delta1, double Delta2);

    void process(LocalVector ldir,const SiPixelRecHit* recHit,
                 std::vector<double>& values);
    void process(LocalVector ldir,const SiStripRecHit2D* recHit,
                 std::vector<double>& values);

    const TrackerGeometry* theTracker;
    double pixelToStripMultiplier, pixelToStripExponent;

    static bool isFirst;
    static float optimalWeight[31][31];
};

#endif
