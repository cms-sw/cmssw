#ifndef Alignment_CommonAlignmentParametrization_ParametersToParametersDerivatives_h
#define Alignment_CommonAlignmentParametrization_ParametersToParametersDerivatives_h

/// \class ParametersToParametersDerivatives
///
/// Class for getting the jacobian d_mother/d_component for various kinds
/// of alignment parametrisations, i.e. the derivatives expressing the influence
/// of the parameters of the 'component' on the parameters of its 'mother'.
/// This is needed e.g. to formulate constraints to remove the additional
/// degrees of freedom introduced if larger structures and their components
/// are aligned simultaneously.
/// The jacobian matrix is
///
/// / dp1_l/dp1_i dp1_l/dp2_i  ...  dp1_l/dpn_i |
/// | dp2_l/dp1_i dp2_l/dp2_i  ...  dp2_l/dpn_i |
/// |      .           .                 .      |
/// |      .           .                 .      |
/// |      .           .                 .      |
/// \ dpm_l/dpm_i dpm_l/dpm_i  ...  dpm_l/dpn_i /
///
/// where 
/// p1_l, p2_l, ..., pn_l are the n parameters of the composite 'mother' object
/// and
/// p1_i, p2_i, ..., pm_i are the m parameters of its component.
///
/// Note that not all combinations of parameters are supported:
/// Please check method isOK() before accessing the derivatives via 
/// operator(unsigned int indParMother, unsigned int indParComp).
///
/// Currently these parameters are supported:
/// - mother: rigid body parameters,
/// - component: rigid body, bowed surface or two bowed surfaces parameters.
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

  /// Return the derivative DeltaParam(mother)/DeltaParam(component).
  /// Indices start with 0 - but check isOK() first!
  /// See class description about matrix.
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
  /// from d(rigid_mother)/d(rigid_component) to d(rigid_mother)/d(bowed_component)
  /// for bad input (length or width zero), set object to invalid: isOK_ = false
  AlgebraicMatrix69 dRigid_dBowed(const AlgebraicMatrix66 &dRigidM2dRigidC,
				  double halfWidth, double halfLength);

  /// data members
  bool            isOK_; /// can we provide the desired?
  TMatrixD derivatives_; /// matrix of derivatives
};

#endif

