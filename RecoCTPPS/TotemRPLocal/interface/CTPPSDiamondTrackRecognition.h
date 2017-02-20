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
#include <map>

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
    typedef std::vector<TF1> hitFunctionVector_t;
    typedef std::map<int,hitFunctionVector_t> hitFunctionVectorMap_t;
    
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

    /// Function for pad efficiency
    hitFunctionVectorMap_t hitFunctionVectorMap_;

};

#endif

