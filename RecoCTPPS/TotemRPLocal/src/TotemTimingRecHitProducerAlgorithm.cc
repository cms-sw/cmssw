/****************************************************************************
*
* This is a part of CTPPS offline software.
* Authors:
*   Laurent Forthomme (laurent.forthomme@cern.ch)
*   Nicola Minafra
*
****************************************************************************/

#include "RecoCTPPS/TotemRPLocal/interface/TotemTimingRecHitProducerAlgorithm.h"

//----------------------------------------------------------------------------------------------------

const double    TotemTimingRecHitProducerAlgorithm::SAMPIC_SAMPLING_PERIOD_NS = 1./7.8;
const double    TotemTimingRecHitProducerAlgorithm::SAMPIC_MAX_NUMBER_OF_SAMPLES = 64;
const double    TotemTimingRecHitProducerAlgorithm::SAMPIC_ADC_V = 1./256;

TotemTimingRecHitProducerAlgorithm::TotemTimingRecHitProducerAlgorithm( const edm::ParameterSet& iConfig ) :
  baselinePoints_               ( iConfig.getParameter<int>( "baselinePoints" ) ),
  risingEdgePointsBeforeTh_     ( iConfig.getParameter<int>( "risingEdgePointsBeforeTh" ) ),
  risingEdgePoints_             ( iConfig.getParameter<int>( "risingEdgePoints" ) ),
  saturationLimit_               ( iConfig.getParameter<double>( "saturationLimit" ) ),
  thresholdFactor_               ( iConfig.getParameter<double>( "thresholdFactor" ) ),
  cfdFraction_                  ( iConfig.getParameter<double>( "cfdFraction" ) ),
  hysteresis_                   ( iConfig.getParameter<double>( "hysteresis" ) )
{}

void
TotemTimingRecHitProducerAlgorithm::build( const CTPPSGeometry* geom, const edm::DetSetVector<TotemTimingDigi>& input, edm::DetSetVector<TotemTimingRecHit>& output )
{
  for ( const auto& vec : input ) {
    const TotemTimingDetId detid( vec.detId() );

    float x_pos = 0,
          y_pos = 0,
          z_pos = 0,
          x_width = 0,
          y_width = 0,
          z_width = 0;

    // retrieve the geometry element associated to this DetID ( if present )
    const DetGeomDesc* det = nullptr;
    try { // no other efficient way to check presence
      det = geom->getSensor( detid );
    }
    catch (cms::Exception&)  { det = nullptr; }

    if ( det != nullptr )
    {
      x_pos = det->translation().x(),
      y_pos = det->translation().y();
      if ( det->parents().empty() )
        edm::LogWarning("TotemTimingRecHitProducerAlgorithm") << "The geometry element for " << detid << " has no parents. Check the geometry hierarchy!";
      else
        z_pos = det->parents()[det->parents().size()-1].absTranslation().z(); // retrieve the plane position;

      x_width = 2.0 * det->params().at( 0 ), // parameters stand for half the size
      y_width = 2.0 * det->params().at( 1 ),
      z_width = 2.0 * det->params().at( 2 );
    }

    edm::DetSet<TotemTimingRecHit>& rec_hits = output.find_or_insert( detid );

    for ( const auto& digi : vec ) {

      // Begin Time of samples computations
      unsigned int offsetOfSamples = digi.getEventInfo().getOffsetOfSamples();
      if ( offsetOfSamples == 0 ) offsetOfSamples = 30; //FW 0 is not sending this, FW > 0 yes
      int cell0TimeClock;
      float cell0TimeInstant;  // Time of first cell
      float triggerCellTimeInstant;    // Time of triggered cell

      unsigned int timestamp = digi.getCellInfo() <= 32 ? digi.getTimestampA() : digi.getTimestampB();

      cell0TimeClock =  timestamp + ( ( digi.getFPGATimestamp() - timestamp ) & 0xFFFFFFF000 ) - digi.getEventInfo().getL1ATimestamp() + digi.getEventInfo().getL1ALatency();
      if ( std::abs( cell0TimeClock ) > 1 )
      {
        std::cout << "cell0TimeClock " << cell0TimeClock << '\n';
        continue;
      }
      if ( cell0TimeClock != 0 ) continue;

      //cell0TimeClock = 0;
      cell0TimeInstant = SAMPIC_MAX_NUMBER_OF_SAMPLES * SAMPIC_SAMPLING_PERIOD_NS * cell0TimeClock;

      if ( digi.getCellInfo() < offsetOfSamples )
        triggerCellTimeInstant = cell0TimeInstant + digi.getCellInfo() * SAMPIC_SAMPLING_PERIOD_NS;
      else
        triggerCellTimeInstant = cell0TimeInstant - ( SAMPIC_MAX_NUMBER_OF_SAMPLES - digi.getCellInfo() ) * SAMPIC_SAMPLING_PERIOD_NS;

      float firstCellTimeInstant = triggerCellTimeInstant - ( SAMPIC_MAX_NUMBER_OF_SAMPLES - offsetOfSamples ) * SAMPIC_SAMPLING_PERIOD_NS;

      // End Time of samples computations

      std::vector<float> data;
      for ( auto it = digi.getSamplesBegin(); it != digi.getSamplesEnd(); ++it )
        data.emplace_back( SAMPIC_ADC_V * (*it) );
      auto max_it = std::max_element(data.begin(), data.end());

      std::vector<float> time( digi.getNumberOfSamples() );
      for ( unsigned int i=0; i<time.size(); ++i )
        time.at(i) = firstCellTimeInstant + i * SAMPIC_SAMPLING_PERIOD_NS;

      float t = -100;

      RegressionResults baselineRegression = SimplifiedLinearRegression( time, data, 0, baselinePoints_ );

      // remove baseline
      std::vector<float> dataCorrected( data.size() );
      for ( unsigned int i=0; i<data.size(); ++i )
      {
        dataCorrected.at(i) = data.at(i) - (baselineRegression.q);
        // std::cout << "rechit: \t ";
        // for (int j=0; j<100*data.at(i); ++j) std::cout << "*";
        // std::cout << '\n';
      }
      auto max_corrected_it = std::max_element(dataCorrected.begin(), dataCorrected.end());

      if ( *max_it < saturationLimit_ ) t = ConstantFractionDiscriminator( time, dataCorrected );

      float tmp_1_ = SmartTimeOfArrival( time, dataCorrected, thresholdFactor_ * baselineRegression.rms );

      // std::vector<float>::const_iterator max = std::max_element(data.begin(), data.end());
      rec_hits.push_back( TotemTimingRecHit( x_pos, x_width, y_pos, y_width, z_pos, z_width, // spatial information
                                             t,
                                             tmp_1_, .0, *max_corrected_it, baselineRegression.rms, mode_));

    }
  }
}

TotemTimingRecHitProducerAlgorithm::RegressionResults TotemTimingRecHitProducerAlgorithm::SimplifiedLinearRegression( const std::vector<float>& time, const std::vector<float>& data, const unsigned int start_at, const unsigned int points ) const
{
  RegressionResults results;
  if ( time.size() != data.size() )
  {
    return results;
  }
  if ( start_at > data.size() )
  {
    return results;
  }
  unsigned int stop_at = std::min( (unsigned int) time.size(), start_at + points );
  unsigned int realPoints = stop_at - start_at;
  auto t_begin = std::next( time.begin(), start_at );
  auto t_end = std::next( time.begin(), stop_at );
  auto d_begin = std::next( data.begin(), start_at );
  auto d_end = std::next( data.begin(), stop_at );

  float sx = .0;
  std::for_each( t_begin, t_end, [&] (float value) { sx += value; } );
  float sxx = .0;
  std::for_each( t_begin, t_end, [&] (float value) { sxx += value*value; } );

  float sy = .0;
  std::for_each( d_begin, d_end, [&] (float value) { sy += value; } );
  float syy = .0;
  std::for_each( d_begin, d_end, [&] (float value) { syy += value*value; } );

  float sxy = .0;
  for ( unsigned int i=0; i<realPoints; ++i )
    sxy += (time.at(i)) * (data.at(i));

  // y = mx + q
  results.m = ( sxy * realPoints - sx * sy ) / ( sxx * realPoints - sx * sx );
  results.q = sy / realPoints - results.m * sx / realPoints;

  float correctedSyy = .0;
  for ( unsigned int i=0; i<realPoints; ++i )
    correctedSyy += pow(data.at(i) - (results.m * time.at(i) + results.q),2);
  results.rms = sqrt( correctedSyy/realPoints );

  return results;
}

int TotemTimingRecHitProducerAlgorithm::FastDiscriminator( const std::vector<float>& data, const float& threshold ) const
{
  int threholdCrossingIndex=-1;
  bool above=false;
  bool lockForHysteresis=false;

  for (unsigned int i=0; i<data.size(); ++i)
  {
    // Look for first edge
    if ( !above && !lockForHysteresis && data.at(i)>threshold )
    {
      threholdCrossingIndex = i;
      above = true;
      lockForHysteresis=true;
    }
    if ( above && lockForHysteresis )        // NOTE: not else if!, "above" can be set in the previous if
    {
      // Lock until above threshold_+hysteresis
      if ( lockForHysteresis && data.at(i)>threshold+hysteresis_ )
      {
        lockForHysteresis=false;
      }
      // Ignore noise peaks
      if ( lockForHysteresis && data.at(i)<threshold )
      {
        above = false;
        lockForHysteresis = false;
        threholdCrossingIndex = -1;      // assigned because of noise
      }
    }
  }

  return threholdCrossingIndex;
}

float TotemTimingRecHitProducerAlgorithm::SmartTimeOfArrival(const std::vector<float>& time, const std::vector<float>& data, const float threshold )
{

  int indexOfThresholdCrossing = FastDiscriminator(data, threshold);

  if ( indexOfThresholdCrossing - risingEdgePointsBeforeTh_ < 1 ) return 0.;

  RegressionResults risingEdgeRegression = SimplifiedLinearRegression( time, data, indexOfThresholdCrossing - risingEdgePointsBeforeTh_, risingEdgePoints_ );

  // Find intersection with zero (baseline subtracted before)
  float t = -100;
  if ( risingEdgeRegression.m > 0.1 && risingEdgeRegression.m < 0.8 ) t = ( .0 - risingEdgeRegression.q / risingEdgeRegression.m );

  mode_ = TotemTimingRecHit::SMART;
  return t;
}

float TotemTimingRecHitProducerAlgorithm::ConstantFractionDiscriminator(const std::vector<float>& time, const std::vector<float>& data )
{
  auto max_it = std::max_element(data.begin(), data.end());
  float max = *max_it;

  float threshold = cfdFraction_ * max;
  int indexOfThresholdCrossing = FastDiscriminator(data, threshold);

  float t=-100;
  //linear interpolation
  if ( indexOfThresholdCrossing-0.5*risingEdgePoints_ >= 0 && indexOfThresholdCrossing+0.5*risingEdgePoints_ < (int) time.size() )
  {
    TGraph rising;
    for (int j=0; j<risingEdgePoints_; ++j) {
      int index = indexOfThresholdCrossing+j-risingEdgePoints_/2;
      rising.SetPoint(j,data.at(index),time.at(index));
    }
    t = rising.Eval(threshold,NULL,"S");
  }

  mode_ = TotemTimingRecHit::CFD;

  return t;

}
