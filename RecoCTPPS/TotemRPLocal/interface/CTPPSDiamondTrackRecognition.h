/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *
 ****************************************************************************/

#ifndef RecoCTPPS_TotemRPLocal_CTPPSDiamondTrackRecognition
#define RecoCTPPS_TotemRPLocal_CTPPSDiamondTrackRecognition

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/TotemRPGeometry.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondLocalTrack.h"

#include "TF1.h"

#define LOWER_HIT_LIMIT_MM -1e3
#define HIGHER_HIT_LIMIT_MM 1e3
#define PAD_FUNCTION (1/(1+exp(-(x-[0])/[2])))*(1/(1+exp((x-[0]-[1])/[2])))

/**
 * \brief Class performing smart reconstruction for CTPPS Diamond Detectors.
**/

class CTPPSDiamondTrackRecognition
{
  public:
    CTPPSDiamondTrackRecognition(const double threshold = 2, const double sigma = 0., const double resolution_mm=0.01);

    ~CTPPSDiamondTrackRecognition();

    void clear();

    void addHit(const CTPPSDiamondRecHit recHit);
    
    void produceTracks(DetSet<CTPPSDiamondLocalTrack> &tracks);
    
  protected:
    const double resolution_mm;

    /// pointer to the geometry
    const TotemRPGeometry* geometry;
    
    /// Function for pad efficiency
    vector<TF1> hit_function_v;

};

#endif

