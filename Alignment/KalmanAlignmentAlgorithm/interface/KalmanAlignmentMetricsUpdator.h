#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentMetricsUpdator_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentMetricsUpdator_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class Alignable;


class KalmanAlignmentMetricsUpdator
{

public:

  KalmanAlignmentMetricsUpdator( const edm::ParameterSet & config ) {}

  virtual ~KalmanAlignmentMetricsUpdator( void ) {}

  virtual void update( const std::vector< Alignable* > & alignables ) = 0;

  virtual const std::vector< Alignable* > additionalAlignables( const std::vector< Alignable* > & alignables ) = 0;

  //virtual const std::map< Alignable*, short int > additionalAlignablesWithDistances( const std::vector< Alignable* > & alignables ) = 0;

  virtual const std::vector< Alignable* > alignables( void ) const = 0;
};


#endif
