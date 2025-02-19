#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentMetricsCalculator_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentMetricsCalculator_h

/// Calculates the metrical distances (stored as short int) for a set of Alignables.
/// See E.Widl, R.Fr\"uhwirth, W.Adam, A Kalman Filter for Track-based Alignment, CMS
/// NOTE-2006/022 for more details.

#include "Alignment/CommonAlignment/interface/Alignable.h"

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "CondFormats/Alignment/interface/Definitions.h"

#include "TTree.h"
#include "TFile.h"


class KalmanAlignmentMetricsCalculator
{

public:

  typedef std::map< Alignable*, short int > SingleDistancesList;
  typedef std::map< Alignable*, SingleDistancesList* > FullDistancesList;

  KalmanAlignmentMetricsCalculator( void );
  ~KalmanAlignmentMetricsCalculator( void );

  /// Update list of distances with a set Alignables.
  void updateDistances( const std::vector< Alignable* >& alignables );

  /// Return map of related Alignables (identified via Alignable*) and their distances
  /// for a distinct Alignable.
  const SingleDistancesList& getDistances( Alignable* i ) const;

  /// Return distance between two Alignables. If there is no metrical
  /// relation between the two Alignables -1 is returned.
  short int operator() ( Alignable* i, Alignable* j ) const;

  /// Set maximum distance to be stored.
  inline void setMaxDistance( short int maxDistance ) { theMaxDistance = maxDistance; }

  /// Number of stored distances.
  unsigned int nDistances( void ) const;

  /// Clear stored distances.
  void clear( void );

  /// Return all known alignables.
  const std::vector< Alignable* > alignables( void ) const;

  void writeDistances( std::string filename );
  void readDistances( std::string filename );

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
  void extractPropagatedDistances( FullDistancesList& changes, Alignable* alignable,
				   SingleDistancesList* oldList, SingleDistancesList* newList );

  /// If the current update of the metric has connected previously unrelated parts (in a metrical sense),
  /// add this information to the table of propagated distances.
  void connect( FullDistancesList& changes, SingleDistancesList* connection,
		Alignable* alignable, short int value );

  void insertDistance( FullDistancesList& dist, Alignable* i, Alignable* j, short int value );
  void insertDistance( SingleDistancesList* distList, Alignable* j, short int value );

  FullDistancesList theDistances;
  short int theMaxDistance;

  SingleDistancesList theDefaultReturnList;

  // For reading and writing

  void createBranches( TTree* tree );
  void setBranchAddresses( TTree* tree );
};


#endif
