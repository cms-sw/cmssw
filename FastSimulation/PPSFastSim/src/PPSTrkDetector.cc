#include "FastSimulation/PPSFastSim/interface/PPSTrkDetector.h"
#include <math.h>
PPSTrkDetector::PPSTrkDetector(double detw, double deth, double detin):
                DetectorWidth(detw), DetectorHeight(deth), DetectorPosition(detin) {clear();}
void PPSTrkDetector::AddHit(double x, double y, double z)
{
// Detector is in the negative side, but DetectorPosition is a positive number
     if (x>0) return; // The detector is on the negative side
     if (fabs(x)>DetectorWidth+DetectorPosition) return; // hit beyond detector area (W)
     if (fabs(x)<DetectorPosition) return;               // hit falls below detector area
     if (fabs(y)>DetectorHeight) return;                 // hit falls beyond detector area (H)
     X.push_back(x);Y.push_back(y);Z.push_back(z);
     NHits++;
}
