#ifndef Alignment_KalmanAlignmentAlgorithm_SingleMetricsUpdator_h
#define Alignment_KalmanAlignmentAlgorithm_SingleMetricsUpdator_h

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsUpdator.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsCalculator.h"

#include <set>


class SimpleMetricsUpdator : public KalmanAlignmentMetricsUpdator
{

public:

  SimpleMetricsUpdator( const edm::ParameterSet & config );

  virtual ~SimpleMetricsUpdator( void ) {}

  virtual void update( const std::vector< Alignable* > & alignables );

  virtual const std::vector< Alignable* > additionalAlignables( const std::vector< Alignable* > & alignables );

  virtual const std::map< Alignable*, short int > additionalAlignablesWithDistances( const std::vector< Alignable* > & alignables );

  virtual const std::vector< Alignable* > alignables( void ) const { return theMetricsCalculator.alignables(); }

private:

  bool additionalSelectionCriterion( Alignable* const& referenceAli,
				     Alignable* const& additionalAli,
				     short int metricalDist ) const;

  KalmanAlignmentMetricsCalculator theMetricsCalculator;

  std::vector< unsigned int > theExcludedSubdetIds;

  bool theASCFlag;
  double theMinDeltaPerp;
  double theMaxDeltaPerp;
  double theMinDeltaZ;
  double theMaxDeltaZ;
  double theGeomDist;
  short int theMetricalThreshold;

};


#endif
