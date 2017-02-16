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

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondLocalTrack.h"

#include "TF1.h"
#include <vector>

/**
 * \brief Class performing smart reconstruction for CTPPS Diamond Detectors.
**/

class CTPPSDiamondTrackRecognition
{
  public:
    CTPPSDiamondTrackRecognition();
    CTPPSDiamondTrackRecognition( const edm::ParameterSet& );

    ~CTPPSDiamondTrackRecognition();

    void clear();

    void addHit(const CTPPSDiamondRecHit recHit);
    
    int produceTracks(edm::DetSet<CTPPSDiamondLocalTrack> &tracks);
    
  protected:
    const double threshold_;
    const double threshold_from_maximum_;
    const double resolution_mm_;
    const double sigma_;
    const double start_from_x_mm_;
    const double stop_at_x_mm_;
    std::string pixel_efficiency_function_;

    /// Function for pad efficiency
    std::vector<TF1> hit_function_v;

};

#endif

