
#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentUserVariables_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentUserVariables_h

#include "Alignment/CommonAlignment/interface/AlignmentUserVariables.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"

#include <string>

/// User variables used by the KalmanAlignmentAlgorithm. The evolution of the estimated alignment
/// parameters is stored in graphs using the DataCollector.

class TrackerTopology;

class KalmanAlignmentUserVariables : public AlignmentUserVariables
{

public:

  /// Create new user variables by specifying the associated Alignable, the Alignable's Id and how
  /// often the evolution of the estimated parameters should be updated.
  KalmanAlignmentUserVariables( Alignable* parent,
                                const TrackerTopology* tTopo,
				int frequency = 100 );

  KalmanAlignmentUserVariables( void ) :
    theParentAlignable( 0 ),
    theNumberOfHits( 0 ),
    theNumberOfUpdates( 0 ),
    theUpdateFrequency( 0 ),
    theFirstUpdate( false ),
    theAlignmentFlag( false )
  {}

  virtual ~KalmanAlignmentUserVariables( void ) {}

  virtual KalmanAlignmentUserVariables* clone( void ) const { return new KalmanAlignmentUserVariables( *this ); }

  /// Return the number of hits.
  inline int numberOfHits( void ) const { return theNumberOfHits; }
  /// Call this function in case the associated Alignable was hit by a particle.
  inline void hit( void ) { ++theNumberOfHits; }

  /// Return the number of updates.
  inline int numberOfUpdates( void ) const { return theNumberOfUpdates; }
  /// Call this function in case the associated Alignable was updated by the alignment algorithm.
  void update( bool enforceUpdate = false );
  /// Update user variables with given alignment parameters.
  void update( const AlignmentParameters* param );
  /// Histogram current estimate of the alignment parameters wrt. the true values.
  void histogramParameters( std::string histoNamePrefix );

  inline const std::string identifier( void ) const { return theIdentifier; }

  inline void setAlignmentFlag( bool flag ) { theAlignmentFlag = flag; }
  inline bool isAligned( void ) const { return theAlignmentFlag; }

  void fixAlignable( void );
  void unfixAlignable( void );

protected:

  const AlgebraicVector extractTrueParameters( void ) const;

  const std::string selectedParameter( const int& selected ) const;
  float selectedScaling( const int& selected ) const;

  const std::string toString( const int& i ) const;

  Alignable* theParentAlignable;

  int theNumberOfHits;
  int theNumberOfUpdates;
  int theUpdateFrequency;

  bool theFirstUpdate;
  bool theAlignmentFlag;

  std::string theIdentifier;
  std::string theTypeAndLayer;

  static const TrackerAlignableId* theAlignableId;
  static const AlignableObjectId* theObjectId;

};


#endif
