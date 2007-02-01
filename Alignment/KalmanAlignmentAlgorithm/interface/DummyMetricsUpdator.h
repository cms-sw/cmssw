#ifndef Alignment_KalmanAlignmentAlgorithm_DummyMetricsUpdator_h
#define Alignment_KalmanAlignmentAlgorithm_DummyMetricsUpdator_h

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsUpdator.h"

#include <set>

/** A dummy metrics updator. It does not calculate the metrics but holds only a list
 *  of all AlignableDets that it so far was updated with.
 */


class DummyMetricsUpdator : public KalmanAlignmentMetricsUpdator
{

public:

  DummyMetricsUpdator( const edm::ParameterSet & config );

  virtual ~DummyMetricsUpdator( void ) {}

  virtual void update( const std::vector< AlignableDet* > & alignableDets );

  virtual const std::vector< AlignableDet* > additionalAlignableDets( const std::vector< AlignableDet* > & alignableDets );

  virtual const std::map< AlignableDet*, short int > additionalAlignableDetsWithDistances( const std::vector< AlignableDet* > & alignableDets );

private:

  std::set< AlignableDet* > theSetOfAllAlignableDets;

  std::vector< unsigned int > theFixedAlignableDetIds;
};


#endif
