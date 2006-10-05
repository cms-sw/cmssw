
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentParameterStore.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/DataCollector.h"

//#include <ext/numeric>

//using namespace __gnu_cxx;
using namespace alignmentservices;


KalmanAlignmentParameterStore::KalmanAlignmentParameterStore( vector < Alignable* > alivec ) :
  AlignmentParameterStore( alivec ), theUpdateCounter( 0 )
{}


AlgebraicMatrix KalmanAlignmentParameterStore::correlations( Alignable* ap1, Alignable* ap2 ) const
{
  bool transpose = false;
  if ( ap2 < ap1 ) transpose = true;

  pair< Alignable*, Alignable* > index = ( transpose == true ) ? make_pair( ap2, ap1 ) : make_pair( ap1, ap2 );
  ExtendedCorrelations::const_iterator itC = theExtCorrelations.find( index );

  if ( itC != theExtCorrelations.end() )
  {
    AlgebraicMatrix corr = ( transpose == true ) ? (*itC).second.first.T() : (*itC).second.first;

    AlgebraicSymMatrix cov1 = ap1->alignmentParameters()->selectedCovariance();
    int nRow = cov1.num_row();
    if ( nRow != corr.num_row() ) { cout << "Inconsistent number of rows" << endl; return AlgebraicMatrix(); }

    AlgebraicSymMatrix cov2 = ap2->alignmentParameters()->selectedCovariance();
    int nCol = cov2.num_col();
    if ( nCol != corr.num_col() ) { cout << "Inconsistent number of columns" << endl; return AlgebraicMatrix(); }

    for ( int iRow = 0; iRow < nRow; iRow++ )
    {
      double factor = sqrt( cov1[iRow][iRow] );
      for ( int iCol = 0; iCol < nCol; iCol++ ) corr[iRow][iCol] *= factor;
    }

    for ( int iCol = 0; iCol < nCol; iCol++ )
    {
      double factor = sqrt( cov2[iCol][iCol] );
      for ( int iRow = 0; iRow < nRow; iRow++ ) corr[iRow][iCol] *= factor;
    }

    return corr;
  }
  else
  {
    return AlgebraicMatrix();
  }
}


void KalmanAlignmentParameterStore::setCorrelations( Alignable* ap1, Alignable* ap2, const AlgebraicMatrix & cov )
{
  pair< Alignable*, Alignable* > index = ( ap1 < ap2 ) ? make_pair( ap1, ap2 ) : make_pair( ap2, ap1 );
  ExtendedCorrelations::iterator itC = theExtCorrelations.find( index );

  if ( itC != theExtCorrelations.end() )
  {
    itC->second.second++;
    static int maxUpdates = 1000; // FIXME: replace hard coded update limit
    if ( itC->second.second > maxUpdates ) return;
  }

  AlgebraicMatrix corr( cov );

  AlgebraicSymMatrix cov1 = ap1->alignmentParameters()->selectedCovariance();
  int nRow = cov1.num_row();
  if ( nRow != corr.num_row() ) { cout << "Inconsistent number of rows" << endl; return; }

  AlgebraicSymMatrix cov2 = ap2->alignmentParameters()->selectedCovariance();
  int nCol = cov2.num_col();
  if ( nCol != corr.num_col() ) { cout << "Inconsistent number of columns" << endl; return; }

  for ( int iRow = 0; iRow < nRow; iRow++ )
  {
    double factor = sqrt( cov1[iRow][iRow] );
    for ( int iCol = 0; iCol < nCol; iCol++ ) corr[iRow][iCol] /= factor;
  }

  for ( int iCol = 0; iCol < nCol; iCol++ )
  {
    double factor = sqrt( cov2[iCol][iCol] );
    for ( int iRow = 0; iRow < nRow; iRow++ ) corr[iRow][iCol] /= factor;
  }

  double r = maxCorrelation( corr );
  // FIXME: replace hard-coded constants.
  double rc = 0.95;
  double wmax = 0.5;

  if ( r >=  1. )
  {
    double weight = wmax/r;
    corr *= weight;
  }
  else if ( r > rc )
  {
    //double weight = 1. - power( ( rc - r )/( rc - 1. ), 3 )*( 1. - wmax );
    double weight = 1. - ( r - rc )/( 1. - rc )*( 1. - wmax );
    corr *= weight;
  }

  bool transpose = false;
  if ( ap2 < ap1 ) { swap( ap1, ap2 ); transpose = true; }

  //theCorrelations[ make_pair( ap1, ap2 ) ] = make_pair( ( transpose ? corr.T() : corr ), theUpdateCounter );

  if ( itC != theExtCorrelations.end() )
  {
    itC->second.first = transpose ? corr.T() : corr;
  }
  else
  {
    theExtCorrelations[ make_pair( ap1, ap2 ) ] = make_pair( ( transpose ? corr.T() : corr ), 1 );
  }

  return;
}


void KalmanAlignmentParameterStore::histogramCorrelations()
{
  if ( DataCollector::isAvailable() )
  {
    for ( ExtendedCorrelations::iterator itC = theExtCorrelations.begin(); itC != theExtCorrelations.end(); itC++ )
    {
      DataCollector::fillHistogram( "corrx", theUpdateCounter, itC->second.first[0][0] );
      DataCollector::fillHistogram( "corrgam", theUpdateCounter, itC->second.first[1][1] );
      DataCollector::fillHistogram( "corrxgam", theUpdateCounter, itC->second.first[0][1] );
      DataCollector::fillHistogram( "corrgamx", theUpdateCounter, itC->second.first[1][0] );
    }
  }
}


void KalmanAlignmentParameterStore::deleteCorrelations( Alignable* ap1, Alignable* ap2 )
{
  pair< Alignable*, Alignable* > index = ( ap1 < ap2 ) ? make_pair( ap1, ap2 ) : make_pair( ap2, ap1 );
  ExtendedCorrelations::iterator itCorrelations = theExtCorrelations.find( index );
  if ( itCorrelations != theExtCorrelations.end() ) theExtCorrelations.erase( itCorrelations );
}


double KalmanAlignmentParameterStore::maxCorrelation( const AlgebraicMatrix & mat ) const
{
  double max = 0;
  double abselement;
  for ( int i = 0; i < mat.num_row(); i++ )
  {
    for ( int j = 0; j < mat.num_col(); j++ )
    {
      abselement = fabs( mat[i][j] );
      if ( abselement > max ) max = abselement;
    }
  }
  return max;
}
