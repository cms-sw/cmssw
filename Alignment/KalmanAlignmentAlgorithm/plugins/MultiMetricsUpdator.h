#ifndef Alignment_KalmanAlignmentAlgorithm_MultipleMetricsUpdator_h
#define Alignment_KalmanAlignmentAlgorithm_MultipleMetricsUpdator_h

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsCalculator.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsUpdator.h"
#include "Alignment/KalmanAlignmentAlgorithm/plugins/SimpleMetricsUpdator.h"

#include <set>


class MultiMetricsUpdator : public KalmanAlignmentMetricsUpdator
{

public:

  MultiMetricsUpdator( const edm::ParameterSet & config );

  virtual ~MultiMetricsUpdator( void );

  virtual void update( const std::vector< Alignable* > & alignables );

  virtual const std::vector< Alignable* > additionalAlignables( const std::vector< Alignable* > & alignables );

  virtual const std::vector< Alignable* > alignables( void ) const;

private:

  std::vector<SimpleMetricsUpdator*> theMetricsUpdators;

};


#endif
