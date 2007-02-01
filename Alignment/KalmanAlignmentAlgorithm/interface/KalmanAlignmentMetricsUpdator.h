#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentMetricsUpdator_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentMetricsUpdator_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class AlignableDet;


class KalmanAlignmentMetricsUpdator
{

public:

  KalmanAlignmentMetricsUpdator( const edm::ParameterSet & config ) {}

  virtual ~KalmanAlignmentMetricsUpdator( void ) {}

  virtual void update( const std::vector< AlignableDet* > & alignableDets ) = 0;

  virtual const std::vector< AlignableDet* > additionalAlignableDets( const std::vector< AlignableDet* > & alignableDets ) = 0;

  virtual const std::map< AlignableDet*, short int > additionalAlignableDetsWithDistances( const std::vector< AlignableDet* > & alignableDets ) = 0;

};


#endif
