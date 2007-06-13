#ifndef Alignment_CommonAlignmentParametrization_CompositeAlignmentParameters_h
#define Alignment_CommonAlignmentParametrization_CompositeAlignmentParameters_h

#include <map>

#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"

/// \class CompositeAlignmentParameters
///
/// Concrete class for 'concatenated' alignment parameters and associated
/// Quantities for a set of Alignables. Provided by AlignmentParameterStore.

class CompositeAlignmentParameters : public AlignmentParameters 
{

public:

  /// vector of alignable components 
  typedef std::vector<Alignable*> Components;

  typedef std::map<AlignableDetOrUnitPtr,Alignable*> AlignableDetToAlignableMap;
  typedef std::map<Alignable*,int> Aliposmap;
  typedef std::map<Alignable*,int> Alilenmap;

  /// constructors 
  CompositeAlignmentParameters() {};
  CompositeAlignmentParameters(const AlgebraicVector& par, const AlgebraicSymMatrix& cov,
			       const Components& comp);

  CompositeAlignmentParameters(const AlgebraicVector& par, const AlgebraicSymMatrix& cov,
			       const Components& comp, const AlignableDetToAlignableMap& alimap,
			       const Aliposmap& aliposmap, const Alilenmap& alilenmap);

  CompositeAlignmentParameters( const DataContainer& data,
				const Components& comp, const AlignableDetToAlignableMap& alimap,
				const Aliposmap& aliposmap, const Alilenmap& alilenmap);

  /// destructor 
  virtual ~CompositeAlignmentParameters();

  /// Clone method (for compatibility with base class)
  CompositeAlignmentParameters* clone( const AlgebraicVector& par, 
				       const AlgebraicSymMatrix& cov) const;

  /// Clone method (for compatibility with base class, same as clone())
  CompositeAlignmentParameters* cloneFromSelected( const AlgebraicVector& par, 
						   const AlgebraicSymMatrix& cov) const;

  /// Clone parameters
  CompositeAlignmentParameters* clone( const AlgebraicVector& par, const AlgebraicSymMatrix& cov, 
				       const AlignableDetToAlignableMap& alimap, 
				       const Aliposmap& aliposmap,
				       const Alilenmap& alilenmap) const;

  /// Clone parameters (same as clone())
  CompositeAlignmentParameters* cloneFromSelected( const AlgebraicVector& par, 
						   const AlgebraicSymMatrix& cov, 
						   const AlignableDetToAlignableMap& alimap,
						   const Aliposmap& aliposmap,
						   const Alilenmap& alilenmap) const;

  /// Get vector of alignable components 
  Components components() const;

  /// Get derivatives 
  AlgebraicMatrix derivatives( const TrajectoryStateOnSurface& tsos, const AlignableDetOrUnitPtr &alidet ) const;
  /// Get derivatives for selected alignables
  AlgebraicMatrix selectedDerivatives( const TrajectoryStateOnSurface& tsos, 
				       const AlignableDetOrUnitPtr &alidet ) const;
  /// for backward compatibility, use std::vector<AlignableDetOrUnitPtr>
  AlgebraicMatrix derivatives( const std::vector<TrajectoryStateOnSurface>& tsosvec, 
			       const std::vector<AlignableDet*>& alidetvec ) const;
  AlgebraicMatrix derivatives( const std::vector<TrajectoryStateOnSurface>& tsosvec,
			       const std::vector<AlignableDetOrUnitPtr>& alidetvec ) const;
  /// for backward compatibility, use std::vector<AlignableDetOrUnitPtr>
  AlgebraicMatrix selectedDerivatives( const std::vector<TrajectoryStateOnSurface> &tsosvec,
				       const std::vector<AlignableDet*> &alidetvec ) const;
  AlgebraicMatrix selectedDerivatives( const std::vector<TrajectoryStateOnSurface> &tsosvec,
 				       const std::vector<AlignableDetOrUnitPtr> &alidetvec ) const;

  /// for backward compatibility, use std::vector<AlignableDetOrUnitPtr>
  AlgebraicVector correctionTerm( const std::vector<TrajectoryStateOnSurface>& tsosvec,
				  const std::vector<AlignableDet*>& alidetvec ) const;
  AlgebraicVector correctionTerm( const std::vector<TrajectoryStateOnSurface>& tsosvec,
 				  const std::vector<AlignableDetOrUnitPtr>& alidetvec ) const;
  /// deprecated due to 'AlignableDet*' interface (legacy code should not be needed anymore)
  AlgebraicMatrix derivativesLegacy ( const TrajectoryStateOnSurface& tsos, 
				      AlignableDet* alidet ) const;
  /// deprecated due to 'AlignableDet*' interface (legacy code should not be needed anymore)
  AlgebraicMatrix selectedDerivativesLegacy( const TrajectoryStateOnSurface& tsos, 
					     AlignableDet* alidet ) const;
  /// deprecated due to 'AlignableDet*' interface (legacy code should not be needed anymore)
  AlgebraicMatrix derivativesLegacy( const std::vector<TrajectoryStateOnSurface>& tsosvec,
				     const std::vector<AlignableDet*>& alidetvec ) const;
  /// deprecated due to 'AlignableDet*' interface (legacy code should not be needed anymore)
  AlgebraicMatrix selectedDerivativesLegacy( const std::vector<TrajectoryStateOnSurface>& tsosvec, 
					     const std::vector<AlignableDet*>& alidetvec ) const;

  /// Get relevant Alignable from AlignableDet 
  Alignable* alignableFromAlignableDet( AlignableDetOrUnitPtr adet ) const;


  /// Extract parameters for subset of alignables
  AlgebraicVector parameterSubset ( const std::vector<Alignable*>& vec ) const;

  /// Extract covariance matrix for subset of alignables
  AlgebraicSymMatrix covarianceSubset( const std::vector<Alignable*>& vec ) const;

  /// Extract covariance matrix elements between two subsets of alignables
  AlgebraicMatrix covarianceSubset ( const std::vector<Alignable*>& veci,
                                     const std::vector<Alignable*>& vecj ) const;

private:

  /// Extract position and length of parameters for a subset of Alignables.
  bool extractPositionAndLength( const std::vector<Alignable*>& alignables,
				 std::vector<int>& posvec,
				 std::vector<int>& lenvec,
				 int& length ) const;

  /// Return vector of alignables without multiple occurences.
  std::vector< Alignable* > extractAlignables( const std::vector< Alignable* >& alignables ) const;

  /// backward compatibility method to convert vectors from specific AlignableDet
  /// to more general AlignableDetOrUnitPtr
  void convert(const std::vector<AlignableDet*> &input,
	       std::vector<AlignableDetOrUnitPtr> &output) const;

  /// Vector of alignable components 
  Components theComponents;

  /// Relate Alignable's and AlignableDet's 
  AlignableDetToAlignableMap theAlignableDetToAlignableMap;

  /// Maps to find parameters/covariance elements for given alignable 
  Aliposmap theAliposmap;
  Alilenmap theAlilenmap;

};

#endif
