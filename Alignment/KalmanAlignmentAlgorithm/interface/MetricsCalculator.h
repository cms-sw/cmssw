#ifndef Alignment_KalmanAlignmentAlgorithm_MetricsCalculator_h
#define Alignment_KalmanAlignmentAlgorithm_MetricsCalculator_h

#include "Alignment/KalmanAlignmentAlgorithm/interface/LookupTable.icc"

/// Calculates the metrics between a set of sensitive modules, which are identified  by a
/// certain index, and stores the result.
/// See E.Widl, R.Fr¨uhwirth, W.Adam, A Kalman Filter for Track-based Alignment, CMS NOTE-
/// 2006/022 for details.


template< class DetIndex >
class MetricsCalculator
{

public:

  MetricsCalculator();
  ~MetricsCalculator();

  void setMaxDistance( int maxDistance ) { theMaxDistance = maxDistance; }
  void computeDistances( vector< DetIndex > theHits );

  map< DetIndex, int > getDistances( DetIndex i ) { return theMetricsTable.getRow( i ); }

  int nElements( void );

  void clear( void ) { theMetricsTable.clear(); }

  int & operator() ( DetIndex i, DetIndex j ) { return theMetricsTable( i, j ); }

protected:

  LookupTable< DetIndex, int > theMetricsTable;

private:

  int theMaxDistance;

};

#endif
