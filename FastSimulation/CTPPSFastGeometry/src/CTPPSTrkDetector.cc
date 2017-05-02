#include "FastSimulation/CTPPSFastGeometry/interface/CTPPSTrkDetector.h"
#include <math.h>
CTPPSTrkDetector::CTPPSTrkDetector(double detw, double deth, double detin):
                ppsDetectorWidth_(detw), ppsDetectorHeight_(deth), ppsDetectorPosition_(detin) {clear();}
void CTPPSTrkDetector::AddHit(unsigned int detID,double x, double y, double z)
{
// Detector is in the negative side, but DetectorPosition is a positive number
    // if (detID >0);
     if (x>0) return; // The detector is on the negative side
     if (fabs(x)>ppsDetectorWidth_+ppsDetectorPosition_) return; // hit beyond detector area (W)
     if (fabs(x)<ppsDetectorPosition_) return;               // hit falls below detector area
     if (fabs(y)>ppsDetectorHeight_*0.5) return;                 // hit falls beyond detector area (H)
     ppsDetId_.push_back(detID);ppsX_.push_back(x);ppsY_.push_back(y);ppsZ_.push_back(z);
     ppsNHits_++;
}
