
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentExtendedCorrelationsEntry.h"


AlignmentExtendedCorrelationsEntry::AlignmentExtendedCorrelationsEntry( void ) :
//   theCounter( 0 ),
  theNRows( 0 ),
  theNCols( 0 )
{}


AlignmentExtendedCorrelationsEntry::AlignmentExtendedCorrelationsEntry( short unsigned int nRows,
									short unsigned int nCols ) :
//   theCounter( 0 ),
  theNRows( nRows ),
  theNCols( nCols ),
  theData( nRows*nCols )
{}


AlignmentExtendedCorrelationsEntry::AlignmentExtendedCorrelationsEntry( short unsigned int nRows,
						      short unsigned int nCols,
						      const float init ) :
//   theCounter( 0 ),
  theNRows( nRows ),
  theNCols( nCols ),
  theData( nRows*nCols, init )
{}


AlignmentExtendedCorrelationsEntry::AlignmentExtendedCorrelationsEntry( const AlgebraicMatrix& mat ) :
//   theCounter( 0 ),
  theNRows( mat.num_row() ),
  theNCols( mat.num_col() ),
  theData( mat.num_row()*mat.num_col() )
{
  for ( int i = 0; i < mat.num_row(); ++i )
  {
    for ( int j = 0; j < mat.num_col(); ++j )
    {
      theData[i*theNCols+j] = mat[i][j];
    }
  }
}


void AlignmentExtendedCorrelationsEntry::operator*=( const float multiply )
{
  for ( std::vector< float >::iterator it = theData.begin(); it != theData.end(); ++it ) (*it) *= multiply;
}


AlgebraicMatrix AlignmentExtendedCorrelationsEntry::matrix( void ) const
{
  AlgebraicMatrix result( theNRows, theNCols );

  for ( int i = 0; i < theNRows; ++i )
  {
    for ( int j = 0; j < theNCols; ++j )
    {
      result[i][j] = theData[i*theNCols+j];
    }
  }

  return result;
}
