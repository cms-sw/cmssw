#ifndef Alignment_CommonAlignmentParametrization_ParametersToParametersDerivatives_h
#define Alignment_CommonAlignmentParametrization_ParametersToParametersDerivatives_h

/// \class ParametersToParametersDerivatives
///
/// Class for calculating derivatives for hierarchies between different kind
/// of alignment parameters (note that not all combinations might be supported!),
/// needed e.g. to formulate constraints to remove the additional degrees of
/// freedom introduced if larger structure and their components are aligned
/// simultaneously.
///
///  $Date: 2010/12/14 01:08:25 $
///  $Revision: 1.2 $
/// (last update by $Author: flucke $)

#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include "TMatrixD.h"

class Alignable;

class ParametersToParametersDerivatives
{
  public:
  ParametersToParametersDerivatives(const Alignable &component, const Alignable &mother);

  /// Indicate whether able to provide the derivatives.
  bool isOK() const { return isOK_;}

  /// Return the derivative DeltaParam(object)/DeltaParam(composedobject), indices start with 0.
  /// But check isOK() first!
  double operator() (unsigned int indParMother, unsigned int indParComp) const; 

  // Not this - would make the internals public:
  //  const TMatrixD& matrix() const { return derivatives_;}

  private:
  /// init by choosing the correct detailed init method depending on parameter types
  bool init(const Alignable &component, int typeComponent,
	    const Alignable &mother,    int typeMother);
  /// init for component and mother both with RigidBody parameters
  bool initRigidRigid(const Alignable &component, const Alignable &mother);
  /// init for component with BowedSurface and mother with RigidBody parameters
  bool initBowedRigid(const Alignable &component, const Alignable &mother);
  /// init for component with TwoBowedSurfaces and mother with RigidBody parameters
  bool init2BowedRigid(const Alignable &component, const Alignable &mother);

  typedef ROOT::Math::SMatrix<double,6,9,ROOT::Math::MatRepStd<double,6,9> > AlgebraicMatrix69;
  AlgebraicMatrix69 dBowed_dRigid(const AlgebraicMatrix66 &f2f,
				  double halfWidth, double halfLength) const;

  /// data members
  bool            isOK_; /// can we provide the desired?
  TMatrixD derivatives_; /// matrix of derivatives
};

#endif

