#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentParameterStore_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentParameterStore_h

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"

#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"

#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"
#include "Alignment/CommonAlignmentParametrization/interface/CompositeAlignmentParameters.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignableData.h"

//#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include <map>

/// Basic class for management of alignment parameters and correlations 

class AlignmentParameterStore 
{

public:

  typedef std::vector<AlignmentParameters*> Parameters;
  typedef std::map< std::pair<Alignable*,Alignable*>,AlgebraicMatrix > Correlations;
  typedef std::vector<Alignable*>   Alignables;
  typedef std::vector<unsigned int> DetIds;

  /// constructor 
  AlignmentParameterStore( std::vector <Alignable*> alivec );

  /// select parameters 
  CompositeAlignmentParameters 
  selectParameters( const std::vector <AlignableDet*>& alignabledets ) const;

  /// update parameters 
  void updateParameters(const CompositeAlignmentParameters& aap);

  /// get all alignables 
  const Alignables& alignables(void) const { return theAlignables; }

  /// get all alignables with valid parameters 
  Alignables validAlignables(void) const;

  /// returns number of alignables 
  int numObjects(void) const { return theAlignables.size(); }

  /// get full correlation map 
  Correlations correlationsMap(void) const { return theCorrelations; }

  /// get number of correlations between alignables 
  int numCorrelations(void) const { return theCorrelations.size(); }

  /// get Alignable which corresponds to a given GeomDet 
  Alignable* alignableFromGeomDet( const GeomDet* geomDet ) const;

  /// get Alignable corresponding to given AlignableDet 
  Alignable* alignableFromAlignableDet( const AlignableDet* alignableDet ) const;

  /// get Alignable corresponding to given DetId
  Alignable* alignableFromDetId(const unsigned int& detId) const;

  /// transform std::vector<TrackingRecHit> into corresponding std::vector<AlignableDet*> 
    //  std::vector<AlignableDet*> alignableDetsFromHits(const std::vector<const TransientTrackingRecHit*>& hitvec);
    // MOVED TO ALIGNABLENAVIGATOR

  /// apply all valid parameters to their alignables 
  void applyParameters(void);

  /// apply parameters of a given alignable 
  void applyParameters(Alignable* alignable);

  /// reset parameters, correlations, user variables 
  void resetParameters(void);

  /// reset parameters of a given alignable 
  void resetParameters(Alignable* ali);

  /// acquire shifts/rotations from alignables of the store and copy into 
  ///  alignment parameters (local frame) 
  void acquireRelativeParameters(void);

  /// apply absolute positions to alignables 
  void applyAlignableAbsolutePositions( const Alignables& alivec, 
										const AlignablePositions& newpos, int& ierr );

  /// apply relative shifts to alignables 
  void applyAlignableRelativePositions( const Alignables& alivec, 
										const AlignableShifts& shifts, int& ierr );

  /// Attach alignment parameters to given alignables 
  void attachAlignmentParameters( const Alignables& alivec, 
								  const Parameters& parvec, int& ierr );

  /// Attach alignment parameters to alignables
  void attachAlignmentParameters(const Parameters& parvec, int& ierr);

  /// Attach correlations to given alignables 
  void attachCorrelations( const Alignables& alivec, const Correlations& cormap, 
						   bool overwrite, int& ierr );

  /// Attach correlations to alignables
  void attachCorrelations( const Correlations& cormap, bool overwrite, int& ierr );

  /// Attach User Variables to given alignables 
  void attachUserVariables( const Alignables& alivec,
							const std::vector<AlignmentUserVariables*>& uvarvec, int& ierr);

  /// Set Alignment position error 
  void setAlignmentPositionError( const Alignables& alivec, double valshift, double valrot );

  /// Obtain type and layer from Alignable 
  std::pair<int,int> typeAndLayer( Alignable* ali );

private:

  // Methods to manage correlation map 
  AlgebraicMatrix correlations(Alignable* ap1, Alignable* ap2) const;

  void setCorrelations(Alignable* ap1, Alignable* ap2, const AlgebraicMatrix& mat);

  // Celper used by constructor to get all DetIds per Alignable
  DetIds findDetIds( Alignable* alignable );

  // data members

  // alignables 
  Alignables theAlignables;

  // correlations 
  Correlations theCorrelations;
 
  // Map of DetIds and Alignables
  typedef  std::map<unsigned int,Alignable*> ActiveAlignablesByDetIdMap;
  ActiveAlignablesByDetIdMap theActiveAlignablesByDetId;

  TrackerAlignableId* theTrackerAlignableId;
  //AlignableNavigator* theNavigator;

};

#endif
