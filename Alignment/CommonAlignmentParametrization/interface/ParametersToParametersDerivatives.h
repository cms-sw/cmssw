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
///  $Date: 2007/10/08 15:56:00 $
///  $Revision: 1.6 $
/// (last update by $Author: cklae $)

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

class Alignable;

class ParametersToParametersDerivatives
{
  public:
  ParametersToParametersDerivatives(const Alignable &component, const Alignable &mother);

  /// Indicate whether able to provide the derivatives.
  bool isOK() const {return isOK_;}

  /// Return the derivative DeltaParam(object)/DeltaParam(composedobject), indices start with 0.
  /// But check isOK() first!
  double operator() (unsigned int indParMother, unsigned int indParComp) const; 

  private:
  bool init(const Alignable &component, int typeComponent,
	    const Alignable &mother,    int typeMother);
  bool initRigidRigid(const Alignable &component, const Alignable &mother);

  /// data members
  bool            isOK_;
  AlgebraicMatrix derivatives_;


};

#endif

