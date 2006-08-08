#ifndef Alignment_CommonAlignmentParametrization_CompositeAlignmentParameters_h
#define Alignment_CommonAlignmentParametrization_CompositeAlignmentParameters_h

#include <map>
#include <vector>

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"

#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"

/// Concrete class for 'concatenated' alignment parameters and associated
/// Quantities for a set of Alignables. Provided by AlignmentParameterStore.

class CompositeAlignmentParameters : public AlignmentParameters 
{

public:

  /// vector of alignable components 
  typedef std::vector<Alignable*> Components;

  typedef std::map<AlignableDet*,Alignable*> AlignableDetToAlignableMap;
  typedef std::map<Alignable*,int> Aliposmap;
  typedef std::map<Alignable*,int> Alilenmap;

  /// constructors 
  CompositeAlignmentParameters() {};
  CompositeAlignmentParameters(const AlgebraicVector& par, 
							   const AlgebraicSymMatrix& cov, const Components& comp);

  CompositeAlignmentParameters(const AlgebraicVector& par, 
							   const AlgebraicSymMatrix& cov, const Components& comp, 
							   const AlignableDetToAlignableMap& map, const Aliposmap& aliposmap,
							   const Alilenmap& alilenmap);

  /// destructor 
  ~CompositeAlignmentParameters();

  /// Clone method (for compatibility with base class)
  CompositeAlignmentParameters* clone( const AlgebraicVector& par, 
									   const AlgebraicSymMatrix& cov) const;

  /// Clone method (for compatibility with base class, same as clone())
  CompositeAlignmentParameters* cloneFromSelected( const AlgebraicVector& par, 
												   const AlgebraicSymMatrix& cov) const;

  /// Clone parameters
  CompositeAlignmentParameters* clone( const AlgebraicVector& par, const AlgebraicSymMatrix& cov, 
									  const AlignableDetToAlignableMap& map, 
									   const Aliposmap& aliposmap,
									   const Alilenmap& alilenmap) const;

  /// Clone parameters (same as clone())
  CompositeAlignmentParameters* cloneFromSelected( const AlgebraicVector& par, 
												   const AlgebraicSymMatrix& cov, 
												   const AlignableDetToAlignableMap& map,
												   const Aliposmap& aliposmap,
												   const Alilenmap& alilenmap) const;

  /// Get vector of alignable components 
  Components components() const;

  /// Get derivatives 
  AlgebraicMatrix derivatives( const TrajectoryStateOnSurface tsos, AlignableDet* alidet ) const;
  /// Get derivatives for selected alignables
  AlgebraicMatrix selectedDerivatives( const TrajectoryStateOnSurface tsos, 
									   AlignableDet* alidet ) const;
  AlgebraicMatrix derivatives( const std::vector<TrajectoryStateOnSurface> tsosvec, 
							   std::vector<AlignableDet*> alidetvec ) const;
  AlgebraicMatrix selectedDerivatives( const std::vector<TrajectoryStateOnSurface> 
									   tsosvec, std::vector<AlignableDet*> alidetvec ) const;

  AlgebraicVector correctionTerm( const std::vector<TrajectoryStateOnSurface> tsosvec,
								  std::vector<AlignableDet*> alidetvec ) const;
  AlgebraicMatrix derivativesLegacy ( const TrajectoryStateOnSurface tsos, 
									  AlignableDet* alidet ) const;
  AlgebraicMatrix selectedDerivativesLegacy( const TrajectoryStateOnSurface tsos, 
											 AlignableDet* alidet ) const;
  AlgebraicMatrix derivativesLegacy( const std::vector<TrajectoryStateOnSurface> tsosvec,
									 std::vector<AlignableDet*> alidetvec ) const;
  AlgebraicMatrix selectedDerivativesLegacy( const std::vector<TrajectoryStateOnSurface> tsosvec, 
											 std::vector<AlignableDet*> alidetvec ) const;

  /// Get relevant Alignable from AlignableDet 
  Alignable* alignableFromAlignableDet( AlignableDet* adet ) const;


  /// Extract parameters for subset of alignables
  AlgebraicVector parameterSubset ( const std::vector<Alignable*>& veci ) const;

  /// Extract covariance between two subsets of alignables
  AlgebraicMatrix covarianceSubset ( const std::vector<Alignable*>& veci,
									 const std::vector<Alignable*>& vecj ) const;

private:

  /// Vector of alignable components 
  Components theComponents;

  /// Relate Alignable's and AlignableDet's 
  AlignableDetToAlignableMap theAlignableDetToAlignableMap;

  /// Maps to find parameters/covariance elements for given alignable 
  Aliposmap theAliposmap;
  Alilenmap theAlilenmap;

};

#endif
