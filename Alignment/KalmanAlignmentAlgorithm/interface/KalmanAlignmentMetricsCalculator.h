#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentMetricsCalculator_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentMetricsCalculator_h

/// Calculates the metrical distances (stored as type Distance) for a set of Alignables,
/// which are identified by a certain index (Index).
/// See E.Widl, R.Fr\"uhwirth, W.Adam, A Kalman Filter for Track-based Alignment, CMS
/// NOTE-2006/022 for more details.

#include "Alignment/CommonAlignment/interface/AlignableDet.h"


class KalmanAlignmentMetricsCalculator
{

public:

  typedef std::map< AlignableDet*, short int > SingleDistancesList;
  typedef std::map< AlignableDet*, SingleDistancesList* > FullDistancesList;

  KalmanAlignmentMetricsCalculator( void );
  ~KalmanAlignmentMetricsCalculator( void );

  /// Update list of distances with a set Alignables.
  void updateDistances( const std::vector< AlignableDet* >& alignables );

  /// Return map of related Alignables (identified via AlignableDet*) and their distances
  /// for a distinct Alignable.
  const SingleDistancesList& getDistances( AlignableDet* i ) const;

  /// Return distance between two Alignables (identified via AlignableDet*). If there is
  /// no metrical relation between the two Alignables -1 is returned.
  short int operator() ( AlignableDet* i, AlignableDet* j ) const;

  /// Set maximum distance to be stored.
  inline void setMaxDistance( short int maxDistance ) { theMaxDistance = maxDistance; }

  /// Number of stored distances.
  unsigned int nDistances( void ) const;

  /// Clear stored distances.
  void clear( void );

private:

  void clearDistances( FullDistancesList& dist );

  /// Update thisList with information from otherList - thisList and otherList are assumed
  /// to belong to different alignables.
  void updateList( SingleDistancesList* thisList, SingleDistancesList* otherList );

  /// Insert changes due to the update of the lists of the current alignables.
  void insertUpdatedDistances( FullDistancesList& updated );

  /// Insert the 'propagated distances' into the lists of the remaining alignables.
  void insertPropagatedDistances( FullDistancesList& propagated );

  /// Extract entries from the updated lists that need to be further propagated.
  void extractPropagatedDistances( FullDistancesList& changes, AlignableDet* alignable,
				   SingleDistancesList* oldList, SingleDistancesList* newList );

  /// If the current update of the metric has connected previously unrelated parts (in a metrical sense),
  /// add this information to the table of propagated distances.
  void connect( FullDistancesList& changes, SingleDistancesList* connection,
		AlignableDet* alignable, short int value );

  void insertDistance( FullDistancesList& dist, AlignableDet* i, AlignableDet* j, short int value );
  void insertDistance( SingleDistancesList* distList, AlignableDet* j, short int value );

  FullDistancesList theDistances;
  short int theMaxDistance;

  SingleDistancesList theDefaultReturnList;
};


#endif
