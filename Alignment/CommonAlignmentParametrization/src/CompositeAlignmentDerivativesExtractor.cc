#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "Alignment/CommonAlignmentParametrization/interface/CompositeAlignmentDerivativesExtractor.h"

//--------------------------------------------------------------------------------------
CompositeAlignmentDerivativesExtractor::
CompositeAlignmentDerivativesExtractor( const std::vector< Alignable* > & alignables,
                                        const std::vector< AlignableDet* > & alignableDets,
                                        const std::vector< TrajectoryStateOnSurface > & tsos )
{
  std::vector<AlignableDetOrUnitPtr> detOrUnits;
  detOrUnits.reserve(alignableDets.size());

  std::vector<AlignableDet*>::const_iterator it, itEnd;
  for (it = alignableDets.begin(), itEnd = alignableDets.end(); it != itEnd; ++it)
    detOrUnits.push_back(AlignableDetOrUnitPtr(*it));

  extractCurrentAlignment( alignables, detOrUnits, tsos );
}

//--------------------------------------------------------------------------------------
CompositeAlignmentDerivativesExtractor::
CompositeAlignmentDerivativesExtractor( const std::vector< Alignable* > & alignables,
                                        const std::vector< AlignableDetOrUnitPtr > & alignableDets,
                                        const std::vector< TrajectoryStateOnSurface > & tsos )
{
  extractCurrentAlignment( alignables, alignableDets, tsos );
}

//--------------------------------------------------------------------------------------

void CompositeAlignmentDerivativesExtractor::
extractCurrentAlignment( const std::vector< Alignable* > & alignables,
                         const std::vector< AlignableDetOrUnitPtr > & alignableDets,
                         const std::vector< TrajectoryStateOnSurface > & tsos )
{

  // sanity check
  if ( alignables.size() != alignableDets.size() )
  {
	edm::LogError("CompositeAlignmentDerivativesExtractor") 
	  << "Inconsistent length of arguments: alignables=" << alignables.size() 
	  << ", alignableDets=" << alignableDets.size();
    return;
  }

  if ( alignables.size() != tsos.size() )
	{
	  edm::LogError("CompositeAlignmentDerivativesExtractor") 
		<< "Inconsistent length of arguments: alignables=" << alignables.size()
		<< ", tsos=" << tsos.size();
	  return;
	}

  std::vector< Alignable* >::const_iterator itAlignable = alignables.begin();
  std::vector< AlignableDetOrUnitPtr >::const_iterator itAlignableDet = alignableDets.begin();
  std::vector< TrajectoryStateOnSurface >::const_iterator itTsos = tsos.begin();

  int nRow = 0;
  int nCollumn = 0;
  unsigned int nAlignables = 0;

  std::vector< AlgebraicMatrix > subDerivatives;
  std::vector< AlgebraicVector > subCorrectionTerm;


  // get the individual derivatives and correction term and determine the dimension
  while ( itAlignable != alignables.end() )
  {
    // Get the current estimate on the alignment parameters
    AlgebraicVector subAlignmentParameters = 
	  ( *itAlignable )->alignmentParameters()->selectedParameters();

    // Get the derivatives or the local coordinates w.r.t. the corresponding alignment parameters
    AlgebraicMatrix subAlignmentDerivatives =
      ( *itAlignable )->alignmentParameters()->selectedDerivatives( *itTsos, *itAlignableDet );

    subDerivatives.push_back( subAlignmentDerivatives.T() );
    subCorrectionTerm.push_back( subAlignmentDerivatives.T()*subAlignmentParameters );

    nRow += 2;
    // check if it is the first occurrence of this Alignable
    if ( count( alignables.begin(), itAlignable, *itAlignable ) == 0 )
    {
	  // matrix is transposed -> num_row() instead of num_col()
      nCollumn += subAlignmentDerivatives.num_row();
      nAlignables++;
    }

    itAlignable++;
    itAlignableDet++;
    itTsos++;
  }

  // construct derivatives and correction term with the right dimension
  theDerivatives = AlgebraicMatrix( nRow, nCollumn, 0 );
  theCorrectionTerm = AlgebraicVector( nRow, 0 );

  if ( alignables.size() == nAlignables )
    // One hit per alignable
    extractWithoutMultipleHits( subCorrectionTerm, subDerivatives );
  else
    // At least one alignable has two hits
    extractWithMultipleHits( subCorrectionTerm, subDerivatives, alignables );

  return;

}

//--------------------------------------------------------------------------------------

void CompositeAlignmentDerivativesExtractor::
extractWithoutMultipleHits( const std::vector< AlgebraicVector > & subCorrectionTerm,
							const std::vector< AlgebraicMatrix > & subDerivatives )
{
  std::vector< AlgebraicVector >::const_iterator itSubCorrectionTerm = subCorrectionTerm.begin();
  std::vector< AlgebraicMatrix >::const_iterator itSubDerivatives = subDerivatives.begin();

  int iRow = 1;
  int iCollumn = 1;

  // Fill in the individual terms
  while ( itSubCorrectionTerm != subCorrectionTerm.end() )
  {
    theCorrectionTerm.sub( iRow, *itSubCorrectionTerm );
    theDerivatives.sub( iRow, iCollumn, *itSubDerivatives );

    iRow += 2;
    iCollumn += ( *itSubDerivatives ).num_col();

    itSubCorrectionTerm++;
    itSubDerivatives++;
  }

  return;
}

//--------------------------------------------------------------------------------------

void CompositeAlignmentDerivativesExtractor::
extractWithMultipleHits( const std::vector< AlgebraicVector > & subCorrectionTerm,
						 const std::vector< AlgebraicMatrix > & subDerivatives,
						 const std::vector< Alignable* > & alignables )
{

  std::vector< AlgebraicVector >::const_iterator itSubCorrectionTerm = subCorrectionTerm.begin();
  std::vector< AlgebraicMatrix >::const_iterator itSubDerivatives = subDerivatives.begin();
  std::vector< Alignable* >::const_iterator itAlignables = alignables.begin();
  std::vector< Alignable* >::const_iterator itPosition;
  std::vector< Alignable* >::const_iterator itLastPosition;

  int iRow = 1;

  // Fill in the individual terms
  while ( itAlignables != alignables.end() )
  {
    theCorrectionTerm.sub( iRow, *itSubCorrectionTerm );

    int iCollumn = 1;
    int iAlignable = 0;

    itLastPosition = find( alignables.begin(), itAlignables, *itAlignables );

    for ( itPosition = alignables.begin(); itPosition != itLastPosition; itPosition++ )
	  {
		if ( count( alignables.begin(), itPosition, *itPosition ) == 0 )
		  iCollumn += subDerivatives[iAlignable].num_col();
		iAlignable++;
	  }

    theDerivatives.sub( iRow, iCollumn, *itSubDerivatives );

    iRow += 2;
	
    itAlignables++;
    itSubCorrectionTerm++;
    itSubDerivatives++;
  }

  return;
}

