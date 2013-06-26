#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentExtendedCorrelationsStore.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentExtendedCorrelationsEntry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/isFinite.h"


AlignmentExtendedCorrelationsStore::AlignmentExtendedCorrelationsStore( const edm::ParameterSet& config )
{
  theMaxUpdates = config.getParameter<int>( "MaxUpdates" );
  theCut = config.getParameter<double>( "CutValue" );
  theWeight = config.getParameter<double>( "Weight" );

  edm::LogInfo("Alignment") << "@SUB=AlignmentExtendedCorrelationsStore::AlignmentExtendedCorrelationsStore"
                            << "Created.";
}


void AlignmentExtendedCorrelationsStore::correlations( Alignable* ap1, Alignable* ap2,
						       AlgebraicSymMatrix& cov, int row, int col ) const
{
  static Alignable* previousAlignable = 0;
  static ExtendedCorrelationsTable* previousCorrelations;

  // Needed by 'resetCorrelations()' to reset the static pointer:
  if ( ap1 == 0 ) { previousAlignable = 0; return; }

  bool transpose = ( ap2 > ap1 );
  if ( transpose ) std::swap( ap1, ap2 ); 

  if ( ap1 == previousAlignable )
  {
    ExtendedCorrelationsTable::const_iterator itC2 = previousCorrelations->find( ap2 );
    if ( itC2 != previousCorrelations->end() )
    {
      transpose ?
	fillCovarianceT( ap1, ap2, (*itC2).second, cov, row, col ) :
	fillCovariance( ap1, ap2, (*itC2).second, cov, row, col );
    }
  }
  else
  {
    ExtendedCorrelations::const_iterator itC1 = theCorrelations.find( ap1 );
    if ( itC1 != theCorrelations.end() )
    {
      previousAlignable = ap1;
      previousCorrelations = (*itC1).second;

      ExtendedCorrelationsTable::const_iterator itC2 = (*itC1).second->find( ap2 );
      if ( itC2 != (*itC1).second->end() )
      {
	transpose ?
	  fillCovarianceT( ap1, ap2, (*itC2).second, cov, row, col ) :
	  fillCovariance( ap1, ap2, (*itC2).second, cov, row, col );
      }
    }
  }

  // don't fill anything into the covariance if there's no entry
  return;
}


void AlignmentExtendedCorrelationsStore::setCorrelations( Alignable* ap1, Alignable* ap2,
							  const AlgebraicSymMatrix& cov, int row, int col )
{
  static Alignable* previousAlignable = 0;
  static ExtendedCorrelationsTable* previousCorrelations;

  // Needed by 'resetCorrelations()' to reset the static pointer:
  if ( ap1 == 0 ) { previousAlignable = 0; return; }

  bool transpose = ( ap2 > ap1 );
  if ( transpose ) std::swap( ap1, ap2 );

  if ( ap1 == previousAlignable )
  {
    fillCorrelationsTable( ap1, ap2, previousCorrelations, cov, row, col, transpose );
  }
  else
  {
    ExtendedCorrelations::iterator itC = theCorrelations.find( ap1 );
    if ( itC != theCorrelations.end() )
    {
      fillCorrelationsTable( ap1, ap2, itC->second, cov, row, col, transpose );
      previousAlignable = ap1;
      previousCorrelations = itC->second;
    }
    else
    {
      // make new entry
      ExtendedCorrelationsTable* newTable = new ExtendedCorrelationsTable;
      fillCorrelationsTable( ap1, ap2, newTable, cov, row, col, transpose );

      theCorrelations[ap1] = newTable;

      previousAlignable = ap1;
      previousCorrelations = newTable;
    }
  }
}


void AlignmentExtendedCorrelationsStore::setCorrelations( Alignable* ap1, Alignable* ap2, AlgebraicMatrix& mat )
{
  bool transpose = ( ap2 > ap1 );
  if ( transpose ) std::swap( ap1, ap2 );

  ExtendedCorrelations::iterator itC1 = theCorrelations.find( ap1 );
  if ( itC1 != theCorrelations.end() )
  { 
    ExtendedCorrelationsTable::iterator itC2 = itC1->second->find( ap1 );
    if ( itC2 != itC1->second->end() )
    {
      itC2->second = transpose ? ExtendedCorrelationsEntry( mat.T() ) : ExtendedCorrelationsEntry( mat );
    }
    else
    {
      (*itC1->second)[ap2] = transpose ? ExtendedCorrelationsEntry( mat.T() ) : ExtendedCorrelationsEntry( mat );
    }
  }
  else
  {
    ExtendedCorrelationsTable* newTable = new ExtendedCorrelationsTable;
    (*newTable)[ap2] = transpose ? ExtendedCorrelationsEntry( mat.T() ) : ExtendedCorrelationsEntry( mat );
    theCorrelations[ap1] = newTable;
  }
}


void AlignmentExtendedCorrelationsStore::getCorrelations( Alignable* ap1, Alignable* ap2, AlgebraicMatrix& mat ) const
{
  bool transpose = ( ap2 > ap1 );
  if ( transpose ) std::swap( ap1, ap2 );

  ExtendedCorrelations::const_iterator itC1 = theCorrelations.find( ap1 );
  if ( itC1 != theCorrelations.end() )
  {
    ExtendedCorrelationsTable::const_iterator itC2 = itC1->second->find( ap2 );
    if ( itC2 != itC1->second->end() )
    {
      mat = transpose ? itC2->second.matrix().T() : itC2->second.matrix();
      return;
    }
  }

  mat = AlgebraicMatrix();
}


bool AlignmentExtendedCorrelationsStore::correlationsAvailable( Alignable* ap1, Alignable* ap2 ) const
{
  bool transpose = ( ap2 > ap1 );
  if ( transpose ) std::swap( ap1, ap2 );

  ExtendedCorrelations::const_iterator itC1 = theCorrelations.find( ap1 );
  if ( itC1 != theCorrelations.end() )
  {
    ExtendedCorrelationsTable::const_iterator itC2 = itC1->second->find( ap2 );
    if ( itC2 != itC1->second->end() ) return true;
  }
  return false;
}


void AlignmentExtendedCorrelationsStore::resetCorrelations( void )
{
  ExtendedCorrelations::iterator itC;
  for ( itC = theCorrelations.begin(); itC != theCorrelations.end(); ++itC ) delete (*itC).second;
  theCorrelations.erase( theCorrelations.begin(), theCorrelations.end() );

  // Reset the static pointers to the 'previous alignables'
  AlgebraicSymMatrix dummy;
  correlations( 0, 0, dummy, 0, 0 );
  setCorrelations( 0, 0, dummy, 0, 0 );
}


unsigned int AlignmentExtendedCorrelationsStore::size( void ) const
{
  unsigned int size = 0;
  ExtendedCorrelations::const_iterator itC;
  for ( itC = theCorrelations.begin(); itC != theCorrelations.end(); ++itC )
    size += itC->second->size();

  return size;
}


void
AlignmentExtendedCorrelationsStore::fillCorrelationsTable( Alignable* ap1, Alignable* ap2,
							   ExtendedCorrelationsTable* table,
							   const AlgebraicSymMatrix& cov,
							   int row, int col, bool transpose )
{
  ExtendedCorrelationsTable::iterator itC = table->find( ap2 );

  if ( itC != table->end() )
  {
    //if ( itC->second.counter() > theMaxUpdates ) return;

    transpose ?
      readFromCovarianceT( ap1, ap2, itC->second, cov, row, col ) :
      readFromCovariance( ap1, ap2, itC->second, cov, row, col );

    //itC->second.incrementCounter();
  }
  else
  {
    int nRow = ap1->alignmentParameters()->numSelected();
    int nCol = ap2->alignmentParameters()->numSelected();
    ExtendedCorrelationsEntry newEntry( nRow, nCol );

    transpose ?
      readFromCovarianceT( ap1, ap2, newEntry, cov, row, col ) :
      readFromCovariance( ap1, ap2, newEntry, cov, row, col );

    (*table)[ap2] = newEntry;
  }
}


void
AlignmentExtendedCorrelationsStore::fillCovariance( Alignable* ap1, Alignable* ap2,  const ExtendedCorrelationsEntry& entry,
						    AlgebraicSymMatrix& cov, int row, int col ) const
{
  int nRow = entry.numRow();
  int nCol = entry.numCol();

  for ( int iRow = 0; iRow < nRow; ++iRow )
  {
    double factor = sqrt(cov[row+iRow][row+iRow]);
    if ( edm::isNotFinite(factor) ) throw cms::Exception("LogicError") << "[AlignmentExtendedCorrelationsStore::fillCovariance] "
							    << "NaN-factor: sqrt(" << cov[row+iRow][row+iRow] << ")";

    for ( int jCol = 0; jCol < nCol; ++jCol )
      cov[row+iRow][col+jCol] = entry( iRow, jCol )*factor;
  }

  for ( int jCol = 0; jCol < nCol; ++jCol )
  {
    double factor = sqrt(cov[col+jCol][col+jCol]);
    if ( edm::isNotFinite(factor) ) throw cms::Exception("LogicError") << "[AlignmentExtendedCorrelationsStore::fillCovariance] "
							    << "NaN-factor: sqrt(" << cov[col+jCol][col+jCol] << ")";

    for ( int iRow = 0; iRow < nRow; ++iRow )
      cov[row+iRow][col+jCol] *= factor;
  }
}


void
AlignmentExtendedCorrelationsStore::fillCovarianceT( Alignable* ap1, Alignable* ap2, const ExtendedCorrelationsEntry& entry,
					     AlgebraicSymMatrix& cov, int row, int col ) const
{
  int nRow = entry.numRow();
  int nCol = entry.numCol();

  for ( int iRow = 0; iRow < nRow; ++iRow )
  {
    double factor = sqrt(cov[col+iRow][col+iRow]);
    if ( edm::isNotFinite(factor) ) throw cms::Exception("LogicError") << "[AlignmentExtendedCorrelationsStore::fillCovarianceT] "
							    << "NaN-factor: sqrt(" << cov[col+iRow][col+iRow] << ")";
    for ( int jCol = 0; jCol < nCol; ++jCol )
      cov[row+jCol][col+iRow] = entry( iRow, jCol )*factor;
  }

  for ( int jCol = 0; jCol < nCol; ++jCol )
  {
    double factor = sqrt(cov[row+jCol][row+jCol]);
    if ( edm::isNotFinite(factor) ) throw cms::Exception("LogicError") << "[AlignmentExtendedCorrelationsStore::fillCovarianceT] "
							    << "NaN-factor: sqrt(" << cov[row+jCol][row+jCol] << ")";
    for ( int iRow = 0; iRow < nRow; ++iRow )
      cov[row+jCol][col+iRow] *= factor;
  }

}


void
AlignmentExtendedCorrelationsStore::readFromCovariance( Alignable* ap1, Alignable* ap2, ExtendedCorrelationsEntry& entry,
							const AlgebraicSymMatrix& cov, int row, int col )
{
  int nRow = entry.numRow();
  int nCol = entry.numCol();

  for ( int iRow = 0; iRow < nRow; ++iRow )
  {
    double factor = sqrt(cov[row+iRow][row+iRow]);
    for ( int jCol = 0; jCol < nCol; ++jCol )
      entry( iRow, jCol ) = cov[row+iRow][col+jCol]/factor;
  }

  double maxCorr = 0;

  for ( int jCol = 0; jCol < nCol; ++jCol )
  {
    double factor = sqrt(cov[col+jCol][col+jCol]);
    for ( int iRow = 0; iRow < nRow; ++iRow )
    {
      entry( iRow, jCol ) /= factor;
      if ( fabs( entry( iRow, jCol ) ) > maxCorr ) maxCorr = fabs( entry( iRow, jCol ) );
    }
  }

  resizeCorruptCorrelations( entry, maxCorr );
}


void
AlignmentExtendedCorrelationsStore::readFromCovarianceT( Alignable* ap1, Alignable* ap2, ExtendedCorrelationsEntry& entry,
							 const AlgebraicSymMatrix& cov, int row, int col )
{
  int nRow = entry.numRow();
  int nCol = entry.numCol();

  for ( int iRow = 0; iRow < nRow; ++iRow )
  {
    double factor = sqrt(cov[col+iRow][col+iRow]);
    for ( int jCol = 0; jCol < nCol; ++jCol )
      entry( iRow, jCol ) = cov[row+jCol][col+iRow]/factor;
  }

  double maxCorr = 0;

  for ( int jCol = 0; jCol < nCol; ++jCol )
  {
    double factor = sqrt(cov[row+jCol][row+jCol]);
    for ( int iRow = 0; iRow < nRow; ++iRow )
    {
      entry( iRow, jCol ) /= factor;
      if ( fabs( entry( iRow, jCol ) ) > maxCorr ) maxCorr = fabs( entry( iRow, jCol ) );
    }
  }

  resizeCorruptCorrelations( entry, maxCorr );
}


void
AlignmentExtendedCorrelationsStore::resizeCorruptCorrelations( ExtendedCorrelationsEntry& entry,
							       double maxCorr )
{
  if ( maxCorr > 1. )
  {
    entry *= theWeight/maxCorr;
  }
  else if ( maxCorr > theCut )
  {
    entry *= 1. - ( maxCorr - theCut )/( 1. - theCut )*( 1. - theWeight );
  }
}
