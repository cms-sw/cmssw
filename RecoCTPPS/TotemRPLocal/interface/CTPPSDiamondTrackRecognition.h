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

#include <vector>
#include <map>
#include "TF1.h"

/**
 * \brief Class performing smart reconstruction for CTPPS Diamond Detectors.
**/

class CTPPSDiamondTrackRecognition
{
  public:
    CTPPSDiamondTrackRecognition( const edm::ParameterSet& );

    ~CTPPSDiamondTrackRecognition();

    void clear();

    void addHit( const CTPPSDiamondRecHit recHit );

    int produceTracks( edm::DetSet<CTPPSDiamondLocalTrack> &tracks );

  private:
    struct HitParameters {
      float center;
      float width;
      HitParameters( const float center, const float width) : center(center), width(width) {};
    };
    typedef std::vector<HitParameters> HitParametersVector;
    typedef std::map<int,HitParametersVector> HitParametersVectorMap;

    static const std::string pixelEfficiencyDefaultFunction_;
    const float threshold_;
    const float thresholdFromMaximum_;
    const float resolution_;
    const float sigma_;
    const float startFromX_;
    const float stopAtX_;
    std::string pixelEfficiencyFunction_;

    float yPosition;
    float yWidth;
    int nameCounter;

    /// Functions for pad efficiency
    TF1 hit_f_;
    HitParametersVectorMap hitParametersVectorMap_;
    std::map<int,int> mhMap_;
};

#endif

