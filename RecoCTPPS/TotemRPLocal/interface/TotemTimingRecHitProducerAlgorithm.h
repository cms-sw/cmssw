/****************************************************************************
*
* This is a part of CTPPS offline software.
* Authors:
*   Laurent Forthomme (laurent.forthomme@cern.ch)
*   Nicola Minafra
*
****************************************************************************/

#ifndef RecoCTPPS_TotemRPLocal_TotemTimingRecHitProducerAlgorithm
#define RecoCTPPS_TotemRPLocal_TotemTimingRecHitProducerAlgorithm

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"
#include "DataFormats/CTPPSDigi/interface/TotemTimingDigi.h"
#include "DataFormats/CTPPSReco/interface/TotemTimingRecHit.h"

#include <algorithm>

#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

class TotemTimingRecHitProducerAlgorithm
{
  public:
    TotemTimingRecHitProducerAlgorithm( const edm::ParameterSet& conf );

    void build( const CTPPSGeometry*, const edm::DetSetVector<TotemTimingDigi>&, edm::DetSetVector<TotemTimingRecHit>& );

    struct RegressionResults{
      float m;
      float q;
      float rms;
      RegressionResults() : m(0), q(0), rms(0) {};
    };

  private:
    static const double SAMPIC_SAMPLING_PERIOD_NS;
    static const double SAMPIC_MAX_NUMBER_OF_SAMPLES;
    static const double SAMPIC_ADC_V;

    unsigned int baselinePoints_;
    unsigned int risingEdgePointsBeforeTh_;
    unsigned int risingEdgePoints_;
    double threholdFactor_;
    double cfdFraction_;
    double hysteresis_;

    float tmp_1_;
    float tmp_2_;

    RegressionResults SimplifiedLinearRegression( const std::vector<float>& time, const std::vector<float>& data, const unsigned int start_at, const unsigned int points ) const;

    int FastDiscriminator( const std::vector<float>& data, const float& threshold ) const;

    float SmartTimeOfArrival(const std::vector<float>& time, const std::vector<float>& data );
    float ConstantFractionDiscriminator(const std::vector<float>& time, const std::vector<float>& data );

};

#endif
