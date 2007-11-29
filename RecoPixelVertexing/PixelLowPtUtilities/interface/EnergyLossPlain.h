#ifndef _EnergyLossPlain_h_
#define _EnergyLossPlain_h_

#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

class SiPixelRecHit;
class SiStripRecHit2D;

using namespace std;

class EnergyLossPlain
{
  public:
    EnergyLossPlain(const TrackerGeometry* theTracker_,
                    double pixelToStripMultiplier,
                    double pixelToStripExponent);
    ~EnergyLossPlain();
    int estimate(const Trajectory* trajectory,
                 vector<pair<int,double> >& arithmeticMean,
                 vector<pair<int,double> >&  truncatedMean);
  private:
    double average (vector<double>& values);
    double truncate(vector<double>& values);
    double logTruncate(vector<double>& values);
    double expected(double Delta1, double Delta2);

    void process(LocalVector ldir,const SiPixelRecHit* recHit,
                 vector<double>& values);
    void process(LocalVector ldir,const SiStripRecHit2D* recHit,
                 vector<double>& values);

    const TrackerGeometry* theTracker;
    double pixelToStripMultiplier, pixelToStripExponent;
};

#endif
