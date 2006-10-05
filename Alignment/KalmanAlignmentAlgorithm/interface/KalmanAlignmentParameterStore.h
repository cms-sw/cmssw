
#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentParameterStore_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentParameterStore_h

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"

using namespace std;

/// Modified AlignmentParameterStore for the KalmanAlignmentAlgorithm. It stores the correlations rather than
/// the off-diagonal elements of the full covariance matrix, i.e. it stores R_ij = C_ij/sqrt(C_ii*C_jj) and
/// not C_ij. In addition it keeps track how often the correlations have been updated and freezes them eventualy.


class KalmanAlignmentParameterStore : public AlignmentParameterStore
{

public:

  typedef map< pair< Alignable*, Alignable* >, pair< AlgebraicMatrix, int > > ExtendedCorrelations;

  KalmanAlignmentParameterStore( vector <Alignable*> alivec );
  virtual ~KalmanAlignmentParameterStore( void ) {}

protected:

  /// methods to manage correlation map
  virtual AlgebraicMatrix correlations( Alignable* ap1, Alignable* ap2 ) const;
  virtual void setCorrelations( Alignable* ap1, Alignable* ap2, const AlgebraicMatrix & mat );

  double maxCorrelation( const AlgebraicMatrix & mat ) const;
  void histogramCorrelations();
  void deleteCorrelations( Alignable* ap1, Alignable* ap2 );

private:

  ExtendedCorrelations theExtCorrelations;

  int theUpdateCounter;

};

#endif
