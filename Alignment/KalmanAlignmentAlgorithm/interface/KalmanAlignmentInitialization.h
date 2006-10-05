#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentInitialization_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentInitialization_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"

#include <string>

using namespace std;

/// Helper class that initializes the Alignable's error matrices and user variables for the
/// KalmanAlignmentAlgorithm. It is also able to produce simple misalignment scenarios.


class KalmanAlignmentInitialization
{

public:

  KalmanAlignmentInitialization( const edm::ParameterSet& config );
  ~KalmanAlignmentInitialization( void );

  void initializeAlignmentParameters( AlignmentParameterStore* store );

private:

  const edm::ParameterSet theConfiguration;

};


#endif
