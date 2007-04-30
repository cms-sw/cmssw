
#ifndef Alignment_CommonAlignmentParametrization_CompositeRigidBodyAlignmentParameters_h
#define Alignment_CommonAlignmentParametrization_CompositeRigidBodyAlignmentParameters_h

#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"

/// \class CompositeRigidBodyAlignmentParameters
///
///  Alignment parameters for 'higher level' object.
///  Derived from RigidBodyAlignmentParameters so that
///  derivatives method can be redefined
///
///  $Date: 2007/03/12 21:28:48 $
///  $Revision: 1.4 $
/// (last update by $Author: cklae $)

class AlignableDetOrUnitPtr;

class CompositeRigidBodyAlignmentParameters : public RigidBodyAlignmentParameters 
{

public:

  /// Constructor
  CompositeRigidBodyAlignmentParameters(Alignable* object, 
					const AlgebraicVector& par, 
					const AlgebraicSymMatrix& cov);

  /// Constructor with selection
  CompositeRigidBodyAlignmentParameters(Alignable* object, 
					const AlgebraicVector& par, 
					const AlgebraicSymMatrix& cov, 
					const std::vector<bool>& sel);

  /// Clone method
  RigidBodyAlignmentParameters* clone( const AlgebraicVector& par, 
				       const AlgebraicSymMatrix& cov) const;
 
  /// Clone method with selection
  RigidBodyAlignmentParameters* cloneFromSelected( const AlgebraicVector& par, 
						   const AlgebraicSymMatrix& cov ) const;

  /// get derivatives
  AlgebraicMatrix derivatives( const TrajectoryStateOnSurface& tsos,
			       const AlignableDetOrUnitPtr &alidet ) const;

};

#endif

