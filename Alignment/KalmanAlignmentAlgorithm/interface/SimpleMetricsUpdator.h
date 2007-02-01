#ifndef Alignment_KalmanAlignmentAlgorithm_MultipleMetricsUpdator_h
#define Alignment_KalmanAlignmentAlgorithm_MultipleMetricsUpdator_h

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsUpdator.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsCalculator.h"

#include <set>


class SimpleMetricsUpdator : public KalmanAlignmentMetricsUpdator
{

public:

  SimpleMetricsUpdator( const edm::ParameterSet & config );

  virtual ~SimpleMetricsUpdator( void ) {}

  virtual void update( const std::vector< AlignableDet* > & alignableDets );

  virtual const std::vector< AlignableDet* > additionalAlignableDets( const std::vector< AlignableDet* > & alignableDets );

  virtual const std::map< AlignableDet*, short int > additionalAlignableDetsWithDistances( const std::vector< AlignableDet* > & alignableDets );

private:

  KalmanAlignmentMetricsCalculator theMetricsCalculator;

  std::vector< unsigned int > theFixedAlignableDetIds;

};


#endif
