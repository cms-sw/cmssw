#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentParameterStore_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentParameterStore_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

#include "Alignment/CommonAlignmentParametrization/interface/CompositeAlignmentParameters.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentCorrelationsStore.h"
// needed for  AlignableShifts, AlignablePositions:
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignableData.h"
#include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"

/// \class AlignmentParameterStore 
///
/// Basic class for management of alignment parameters and correlations 
///
///  $Date: 2007/04/30 12:11:37 $
///  $Revision: 1.11 $
/// (last update by $Author: flucke $)

class GeomDet;
class Alignable;
class AlignableDet;
class TrackerAlignableId;

class AlignmentParameterStore 
{

public:

  typedef std::vector<AlignmentParameters*> Parameters;
  typedef std::map< std::pair<Alignable*,Alignable*>,AlgebraicMatrix > Correlations;
  typedef std::vector<Alignable*>   Alignables;
  typedef std::vector<unsigned int> DetIds;

  /// constructor 
  AlignmentParameterStore( const Alignables &alis, const edm::ParameterSet& config );

  /// destructor 
  virtual ~AlignmentParameterStore();

  /// select parameters
  /// (for backward compatibility, use with vector<AlignableDetOrUnitPtr> as argument instead)
  CompositeAlignmentParameters
    selectParameters( const std::vector <AlignableDet*>& alignabledets ) const;
  /// select parameters 
  CompositeAlignmentParameters
    selectParameters( const std::vector <AlignableDetOrUnitPtr>& alignabledets ) const;
  /// select parameters 
  CompositeAlignmentParameters
    selectParameters( const std::vector <Alignable*>& alignables ) const;

  /// update parameters 
  void updateParameters(const CompositeAlignmentParameters& aap);

  /// get all alignables 
  const Alignables& alignables(void) const { return theAlignables; }

  /// get all alignables with valid parameters 
  Alignables validAlignables(void) const;

  /// returns number of alignables 
  int numObjects(void) const { return theAlignables.size(); }

  /// get full correlation map 
  AlignmentCorrelationsStore* correlationsStore( void ) const { return theCorrelationsStore; }

  /// get number of correlations between alignables 
  const unsigned int numCorrelations( void ) const { return theCorrelationsStore->size(); }

  /// Obsolete: Use AlignableNavigator::alignableDetFromGeomDet and alignableFromAlignableDet
/*   Alignable* alignableFromGeomDet( const GeomDet* geomDet ) const; */

  /// get Alignable corresponding to given AlignableDet (non-const ref. argument since might be returned)
  Alignable* alignableFromAlignableDet( AlignableDetOrUnitPtr alignableDet ) const;

  /// Obsolete: Use AlignableNavigator::alignableDetFromDetId and alignableFromAlignableDet
/*   Alignable* alignableFromDetId(const unsigned int& detId) const; */

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
  void applyAlignableAbsolutePositions( const Alignables& alis, 
                                        const AlignablePositions& newpos, int& ierr );

  /// apply relative shifts to alignables 
  void applyAlignableRelativePositions( const Alignables& alivec, 
                                        const AlignableShifts& shifts, int& ierr );

  /// Attach alignment parameters to given alignables 
  void attachAlignmentParameters( const Alignables& alivec, const Parameters& parvec, int& ierr );

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
  std::pair<int,int> typeAndLayer( const Alignable* ali ) const;

  /// a single alignable parameter of an Alignable
  typedef std::pair<Alignable*, unsigned int> ParameterId;
  /// Assuming aliMaster has (sub-)components aliComps with parameters
  /// (cf. Alignable::firstParamComponents), paramIdsVecOut and factorsVecOut will be filled
  /// (in parallel) with constraints on the alignment parameters of aliMaster to get rid of the
  /// addionally introduced degrees of freedom:
  /// The 'vector product' of the parameters identified by ParameterId in std::vector<ParameterId>
  /// and the factors in std::vector<float> has to vanish (i.e. == 0.),
  /// |factor| < epsilon will be treated as 0.
  bool hierarchyConstraints(const Alignable *aliMaster, const Alignables &aliComps,
			    std::vector<std::vector<ParameterId> > &paramIdsVecOut,
			    std::vector<std::vector<float> > &factorsVecOut,
			    float epsilon = 1.e-15) const;

protected:

  // storage for correlations
  AlignmentCorrelationsStore* theCorrelationsStore;

private:
  // data members

  /// alignables 
  Alignables theAlignables;

  TrackerAlignableId* theTrackerAlignableId;

};

#endif
