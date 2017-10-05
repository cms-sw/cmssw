#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentCorrelationsStore.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


AlignmentCorrelationsStore::AlignmentCorrelationsStore( void )
{
  edm::LogInfo("Alignment") << "@SUB=AlignmentCorrelationsStore::AlignmentCorrelationsStore "
                            << "\nCreated.";
}


void AlignmentCorrelationsStore::correlations( Alignable* ap1, Alignable* ap2,
					       AlgebraicSymMatrix& cov, int row, int col ) const
{
  static Alignable* previousAlignable = nullptr;
  static CorrelationsTable* previousCorrelations;

  // Needed by 'resetCorrelations()' to reset the static pointer:
  if ( ap1 == nullptr ) { previousAlignable = nullptr; return; }

  bool transpose = ( ap2 > ap1 );
  if ( transpose ) std::swap( ap1, ap2 ); 

  if ( ap1 == previousAlignable )
  {
    CorrelationsTable::const_iterator itC2 = previousCorrelations->find( ap2 );
    if ( itC2 != previousCorrelations->end() )
    {
      transpose ?
	fillCovarianceT( ap1, ap2, (*itC2).second, cov, row, col ) :
	fillCovariance( ap1, ap2, (*itC2).second, cov, row, col );
    }
  }
  else
  {
    Correlations::const_iterator itC1 = theCorrelations.find( ap1 );
    if ( itC1 != theCorrelations.end() )
    {
      previousAlignable = ap1;
      previousCorrelations = (*itC1).second;

      CorrelationsTable::const_iterator itC2 = (*itC1).second->find( ap2 );
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


void AlignmentCorrelationsStore::setCorrelations( Alignable* ap1, Alignable* ap2,
						  const AlgebraicSymMatrix& cov, int row, int col )
{
  static Alignable* previousAlignable = nullptr;
  static CorrelationsTable* previousCorrelations;

  // Needed by 'resetCorrelations()' to reset the static pointer:
  if ( ap1 == nullptr ) { previousAlignable = nullptr; return; }

  bool transpose = ( ap2 > ap1 );
  if ( transpose ) std::swap( ap1, ap2 );

  if ( ap1 == previousAlignable )
  {
    fillCorrelationsTable( ap1, ap2, previousCorrelations, cov, row, col, transpose );
  }
  else
  {
    Correlations::iterator itC = theCorrelations.find( ap1 );
    if ( itC != theCorrelations.end() )
    {
      fillCorrelationsTable( ap1, ap2, itC->second, cov, row, col, transpose );
      previousAlignable = ap1;
      previousCorrelations = itC->second;
    }
    else
    {
      CorrelationsTable* newTable = new CorrelationsTable;
      fillCorrelationsTable( ap1, ap2, newTable, cov, row, col, transpose );

      theCorrelations[ap1] = newTable;
      previousAlignable = ap1;
      previousCorrelations = newTable;
    }
  }
}


void AlignmentCorrelationsStore::setCorrelations( Alignable* ap1, Alignable* ap2, AlgebraicMatrix& mat )
{
  bool transpose = ( ap2 > ap1 );
  if ( transpose ) std::swap( ap1, ap2 );

  Correlations::iterator itC1 = theCorrelations.find( ap1 );
  if ( itC1 != theCorrelations.end() )
  {
    (*itC1->second)[ap2] = transpose ? mat.T() : mat;
  }
  else
  {
    CorrelationsTable* newTable = new CorrelationsTable;
    (*newTable)[ap2] = transpose ? mat.T() : mat;
    theCorrelations[ap1] = newTable;
  }
}


bool AlignmentCorrelationsStore::correlationsAvailable( Alignable* ap1, Alignable* ap2 ) const
{
  bool transpose = ( ap2 > ap1 );
  if ( transpose ) std::swap( ap1, ap2 );

  Correlations::const_iterator itC1 = theCorrelations.find( ap1 );
  if ( itC1 != theCorrelations.end() )
  {
    CorrelationsTable::const_iterator itC2 = itC1->second->find( ap2 );
    if ( itC2 != itC1->second->end() ) return true;
  }
  return false;
}


void AlignmentCorrelationsStore::resetCorrelations( void )
{
  Correlations::iterator itC;
  for ( itC = theCorrelations.begin(); itC != theCorrelations.end(); ++itC ) delete (*itC).second;
  theCorrelations.erase( theCorrelations.begin(), theCorrelations.end() );

  // Reset the static pointers to the 'previous alignables'
  AlgebraicSymMatrix dummy;
  correlations( nullptr, nullptr, dummy, 0, 0 );
  setCorrelations( nullptr, nullptr, dummy, 0, 0 );
}


unsigned int AlignmentCorrelationsStore::size( void ) const
{
  unsigned int size = 0;
  Correlations::const_iterator itC;
  for ( itC = theCorrelations.begin(); itC != theCorrelations.end(); ++itC )
    size += itC->second->size();

  return size;
}


void
AlignmentCorrelationsStore::fillCorrelationsTable( Alignable* ap1, Alignable* ap2, CorrelationsTable* table,
						   const AlgebraicSymMatrix& cov, int row, int col, bool transpose )
{
  CorrelationsTable::iterator itC = table->find( ap2 );

  if ( itC != table->end() )
  {
    transpose ?
      readFromCovarianceT( ap1, ap2, itC->second, cov, row, col ) :
      readFromCovariance( ap1, ap2, itC->second, cov, row, col );
  }
  else
  {
    int nRow = ap1->alignmentParameters()->numSelected();
    int nCol = ap2->alignmentParameters()->numSelected();
    AlgebraicMatrix newEntry( nRow, nCol );

    transpose ?
      readFromCovarianceT( ap1, ap2, newEntry, cov, row, col ) :
      readFromCovariance( ap1, ap2, newEntry, cov, row, col );

    (*table)[ap2] = newEntry;
  }
}


void
AlignmentCorrelationsStore::fillCovariance( Alignable* ap1, Alignable* ap2, const AlgebraicMatrix& entry,
					    AlgebraicSymMatrix& cov, int row, int col ) const
{
  int nRow = entry.num_row();
  int nCol = entry.num_col();

  for ( int iRow = 0; iRow < nRow; ++iRow )
    for ( int jCol = 0; jCol < nCol; ++jCol )
      cov[row+iRow][col+jCol] = entry[iRow][jCol];
}


void
AlignmentCorrelationsStore::fillCovarianceT( Alignable* ap1, Alignable* ap2, const AlgebraicMatrix& entry,
					     AlgebraicSymMatrix& cov, int row, int col ) const
{
  int nRow = entry.num_row();
  int nCol = entry.num_col();

  for ( int iRow = 0; iRow < nRow; ++iRow )
    for ( int jCol = 0; jCol < nCol; ++jCol )
      cov[row+jCol][col+iRow] = entry[iRow][jCol];
}


void
AlignmentCorrelationsStore::readFromCovariance( Alignable* ap1, Alignable* ap2, AlgebraicMatrix& entry,
						const AlgebraicSymMatrix& cov, int row, int col )
{
  int nRow = entry.num_row();
  int nCol = entry.num_col();

  for ( int iRow = 0; iRow < nRow; ++iRow )
    for ( int jCol = 0; jCol < nCol; ++jCol )
      entry[iRow][jCol] = cov[row+iRow][col+jCol];
}


void
AlignmentCorrelationsStore::readFromCovarianceT( Alignable* ap1, Alignable* ap2, AlgebraicMatrix& entry,
						 const AlgebraicSymMatrix& cov, int row, int col )
{
  int nRow = entry.num_row();
  int nCol = entry.num_col();

  for ( int iRow = 0; iRow < nRow; ++iRow )
    for ( int jCol = 0; jCol < nCol; ++jCol )
      entry[iRow][jCol] = cov[row+jCol][col+iRow];
}
