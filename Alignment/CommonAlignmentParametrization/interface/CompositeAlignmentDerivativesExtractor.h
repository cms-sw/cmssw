#ifndef Alignment_CommonAlignmentParametrization_CompositeAlignmentDerivativesExtractor_H
#define Alignment_CommonAlignmentParametrization_CompositeAlignmentDerivativesExtractor_H

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"

/// A helper class to extract derivatives from composite alignable objects

class CompositeAlignmentDerivativesExtractor
{

public:
  
  /// constructor
  CompositeAlignmentDerivativesExtractor( const std::vector< Alignable* > & alignables,
										  const std::vector< AlignableDet* > & alignableDets,
										  const std::vector< TrajectoryStateOnSurface > & tsos );

  /// destructor
  ~CompositeAlignmentDerivativesExtractor( void ) {};

  const AlgebraicMatrix & derivatives( void ) const { return theDerivatives; }
  const AlgebraicVector & correctionTerm( void ) const { return theCorrectionTerm; }
  
private:
  
  void extractCurrentAlignment( const std::vector< Alignable* > & alignables,
								const std::vector< AlignableDet* > & alignableDets,
								const std::vector< TrajectoryStateOnSurface > & tsos );
  
  void extractWithoutMultipleHits( const std::vector< AlgebraicVector > & subCorrectionTerm,
								   const std::vector< AlgebraicMatrix > & subDerivatives );
  
  void extractWithMultipleHits( const std::vector< AlgebraicVector > & subCorrectionTerm,
								const std::vector< AlgebraicMatrix > & subDerivatives,
								const std::vector< Alignable* > & alignables );
  
  AlgebraicMatrix theDerivatives;
  AlgebraicVector theCorrectionTerm;
  
};

#endif
