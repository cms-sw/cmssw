#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentParameterStore_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentParameterStore_h

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignableData.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentCorrelationsStore.h"
#include "Alignment/CommonAlignmentParametrization/interface/CompositeAlignmentParameters.h"

/// \class AlignmentParameterStore 
///
/// Basic class for management of alignment parameters and correlations 
///
///  $Date: 2013/01/07 20:56:25 $
///  $Revision: 1.20 $
/// (last update by $Author: wmtan $)

namespace edm { class ParameterSet; }
class AlignmentUserVariables;
class TrackerTopology;

class AlignmentParameterStore 
{

public:

  typedef std::vector<AlignmentParameters*> Parameters;
  typedef std::map< std::pair<Alignable*,Alignable*>,AlgebraicMatrix > Correlations;
  typedef std::vector<unsigned int> DetIds;

  /// constructor 
  AlignmentParameterStore( const align::Alignables &alis, const edm::ParameterSet& config );

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
  void updateParameters(const CompositeAlignmentParameters& aap, bool updateCorrelations = true);

  /// get all alignables 
  const align::Alignables& alignables(void) const { return theAlignables; }

  /// get all alignables with valid parameters 
  align::Alignables validAlignables(void) const;

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

  /// cache the current position, rotation and other parameters
  void cacheTransformations(void);

  /// restore the previously cached position, rotation and other parameters
  void restoreCachedTransformations(void);

  /// acquire shifts/rotations from alignables of the store and copy into 
  ///  alignment parameters (local frame) 
  void acquireRelativeParameters(void);

  /// apply absolute positions to alignables 
  void applyAlignableAbsolutePositions( const align::Alignables& alis, 
                                        const AlignablePositions& newpos, int& ierr );

  /// apply relative shifts to alignables 
  void applyAlignableRelativePositions( const align::Alignables& alivec, 
                                        const AlignableShifts& shifts, int& ierr );

  /// Attach alignment parameters to given alignables 
  void attachAlignmentParameters( const align::Alignables& alivec, const Parameters& parvec, int& ierr );

  /// Attach alignment parameters to alignables
  void attachAlignmentParameters(const Parameters& parvec, int& ierr);

  /// Attach correlations to given alignables 
  void attachCorrelations( const align::Alignables& alivec, const Correlations& cormap, 
                           bool overwrite, int& ierr );

  /// Attach correlations to alignables
  void attachCorrelations( const Correlations& cormap, bool overwrite, int& ierr );

  /// Attach User Variables to given alignables 
  void attachUserVariables( const align::Alignables& alivec,
                            const std::vector<AlignmentUserVariables*>& uvarvec, int& ierr);

  /// Set Alignment position error 
  void setAlignmentPositionError( const align::Alignables& alivec, double valshift, double valrot );

  /// Obtain type and layer from Alignable 
  std::pair<int,int> typeAndLayer( const Alignable* ali, const TrackerTopology* tTopo ) const;

  /// a single alignable parameter of an Alignable
  typedef std::pair<Alignable*, unsigned int> ParameterId;
  /// Assuming aliMaster has (sub-)components aliComps with alignment parameters
  /// (cf. Alignable::firstParamComponents), paramIdsVecOut and factorsVecOut will be filled
  /// (in parallel) with constraints on the selected alignment parameters of aliMaster to
  /// get rid of the additionally introduced degrees of freedom:
  /// The 'vector product' of the parameters identified by ParameterId in std::vector<ParameterId>
  /// and the factors in std::vector<double> has to vanish (i.e. == 0.),
  /// |factor| < epsilon will be treated as 0.
  /// If all == false, skip constraint on aliMaster's degree of freedom 'i' if 'i'th alignment
  /// parameter of aliMaster is not selected, i.e. constrain only for otherwise doubled d.o.f.
  /// If all == true, produce one constraint for each of the aliMaster's parameters
  /// irrespective of whether they are selecte dor not.
  /// paramIdsVecOut and factorsVecOut contain exactly one std::vector per selected alignment
  /// parameter of aliMaster, but in principle these can be empty...
  /// Note that not all combinations of AlignmentParameters classes ar supported.
  /// If not there will be an exception (and false returned...)
  bool hierarchyConstraints(const Alignable *aliMaster, const align::Alignables &aliComps,
			    std::vector<std::vector<ParameterId> > &paramIdsVecOut,
			    std::vector<std::vector<double> > &factorsVecOut,
			    bool all, double epsilon) const;

protected:

  // storage for correlations
  AlignmentCorrelationsStore* theCorrelationsStore;

private:
  // data members

  /// alignables 
  align::Alignables theAlignables;

};

#endif
