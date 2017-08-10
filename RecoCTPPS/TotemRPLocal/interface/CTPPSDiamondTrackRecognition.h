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
#include <unordered_map>
#include "TF1.h"

/**
 * \brief Class performing smart reconstruction for CTPPS Diamond Detectors.
 * \date Jan 2017
**/
class CTPPSDiamondTrackRecognition
{
  public:
    CTPPSDiamondTrackRecognition( const edm::ParameterSet& );
    ~CTPPSDiamondTrackRecognition();

    /// Reset the list of hits
    void clear();

    /// Feed a new hit to the tracks recognition algorithm
    void addHit( const CTPPSDiamondRecHit& recHit );

    /// Produce a collection of tracks for the current station, given its hits collection
    int produceTracks( edm::DetSet<CTPPSDiamondLocalTrack>& tracks );

  private:
    struct HitParameters {
      HitParameters( const float center, const float width ) :
        center( center ), width( width ) {}
      float center;
      float width;
    };
    typedef std::vector<HitParameters> HitParametersVector;
    typedef std::unordered_map<int,HitParametersVector> HitParametersVectorMap;

    /// Default hit function accounting for the pad spatial efficiency
    static const std::string pixelEfficiencyDefaultFunction_;

    const float threshold_;
    const float thresholdFromMaximum_;
    const float resolution_;
    const float sigma_;
    const float startFromX_;
    const float stopAtX_;

    float yPosition_;
    float yWidth_;
    float yPositionInitial_;
    float yWidthInitial_;

    /// Function for pad efficiency
    TF1 hit_f_;
    HitParametersVectorMap hitParametersVectorMap_;
    std::unordered_map<int,int> mhMap_;
};

#endif
