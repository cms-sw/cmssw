#include "FastSimulation/CTPPSFastGeometry/interface/CTPPSTrkDetector.h"
#include <math.h>
CTPPSTrkDetector::CTPPSTrkDetector(double detw, double deth, double detin):
                DetectorWidth(detw), DetectorHeight(deth), DetectorPosition(detin) {clear();}
void CTPPSTrkDetector::AddHit(unsigned int detID,double x, double y, double z)
{
// Detector is in the negative side, but DetectorPosition is a positive number
    // if (detID >0);
     if (x>0) return; // The detector is on the negative side
     if (fabs(x)>DetectorWidth+DetectorPosition) return; // hit beyond detector area (W)
     if (fabs(x)<DetectorPosition) return;               // hit falls below detector area
     if (fabs(y)>DetectorHeight/2.) return;                 // hit falls beyond detector area (H)
     DetId.push_back(detID);X.push_back(x);Y.push_back(y);Z.push_back(z);
     NHits++;
}
