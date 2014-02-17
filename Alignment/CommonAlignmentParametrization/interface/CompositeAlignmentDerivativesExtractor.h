#ifndef Alignment_CommonAlignmentParametrization_CompositeAlignmentDerivativesExtractor_H
#define Alignment_CommonAlignmentParametrization_CompositeAlignmentDerivativesExtractor_H

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/// \class CompositeAlignmentDerivativesExtractor
///
/// A helper class to extract derivatives from composite alignable objects
///
///  $Date: 2007/10/08 15:55:59 $
///  $Revision: 1.8 $
/// (last update by $Author: cklae $)

class Alignable;
class AlignableDet;
class AlignableDetOrUnitPtr;
class TrajectoryStateOnSurface;

class CompositeAlignmentDerivativesExtractor
{

public:
  
  /// deprecated  constructor for backward compatibility (use mor general AlignableDetOrUnitPtr)
  CompositeAlignmentDerivativesExtractor( const std::vector< Alignable* > & alignables,
					  const std::vector< AlignableDet* > & alignableDets,
					  const std::vector< TrajectoryStateOnSurface > & tsos );
  /// constructor
  CompositeAlignmentDerivativesExtractor( const std::vector< Alignable* > & alignables,
					  const std::vector< AlignableDetOrUnitPtr > & alignableDets,
					  const std::vector< TrajectoryStateOnSurface > & tsos );

  /// destructor
  ~CompositeAlignmentDerivativesExtractor( void ) {};

  const AlgebraicMatrix & derivatives( void ) const { return theDerivatives; }
  const AlgebraicVector & correctionTerm( void ) const { return theCorrectionTerm; }
  
private:
  
  void extractCurrentAlignment( const std::vector< Alignable* > & alignables,
				const std::vector< AlignableDetOrUnitPtr > & alignableDets,
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
