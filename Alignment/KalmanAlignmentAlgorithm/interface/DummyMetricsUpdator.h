#ifndef Alignment_KalmanAlignmentAlgorithm_DummyMetricsUpdator_h
#define Alignment_KalmanAlignmentAlgorithm_DummyMetricsUpdator_h

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsUpdator.h"

#include <set>

/** A dummy metrics updator. It does not calculate the metrics but holds only a list
 *  of all Alignables that it so far was updated with.
 */


class DummyMetricsUpdator : public KalmanAlignmentMetricsUpdator
{

public:

  DummyMetricsUpdator( const edm::ParameterSet & config );

  virtual ~DummyMetricsUpdator( void ) {}

  virtual void update( const std::vector< Alignable* > & alignables );

  virtual const std::vector< Alignable* > additionalAlignables( const std::vector< Alignable* > & alignables );

  virtual const std::map< Alignable*, short int > additionalAlignablesWithDistances( const std::vector< Alignable* > & alignables );

  virtual const std::vector< Alignable* > alignables( void ) const;

private:

  std::set< Alignable* > theSetOfAllAlignables;

  std::vector< unsigned int > theFixedAlignableIds;
};


#endif
