
#ifndef Alignment_CommonAlignment_ALIGNMENTPARAMETERS_H
#define Alignment_CommonAlignment_ALIGNMENTPARAMETERS_H

#include "CommonDet/DetGeometry/interface/AlgebraicObjects.h"

#include "CommonReco/DetAlignment/interface/AlignmentUserVariables.h"

/** Base class for alignment parameters 
 *  It contains a parameter vector of size N and symmetric covariance 
 *  matrix of size N. There is a pointer to the Alignable to
 *  which the parameters belong. There is also a pointer to
 *  UserVariables. Parameters can be enabled/disabled using theSelector. 
 *  bValid declares if the parameters are 'valid'.  
 *  The methods *Selected* set/return only the active
 *  parameters/derivatives/covariance as subvector/submatrix
 *  of reduced size.
 */

class Alignable;
class AlignableDet;
class TrajectoryStateOnSurface;
class RecHit;

class AlignmentParameters {

  public:

  /** constructors */
  AlignmentParameters();
  AlignmentParameters(Alignable* object, const AlgebraicVector& par, 
                      const AlgebraicSymMatrix& cov);
  AlignmentParameters(Alignable* object, const AlgebraicVector& par, 
                      const AlgebraicSymMatrix& cov, const vector<bool>& sel);

  /** destructor */
  virtual ~AlignmentParameters();

  /** enforce clone methods in derived classes */
  virtual AlignmentParameters* clone(const AlgebraicVector& par,
			     const AlgebraicSymMatrix& cov) const=0;
  virtual AlignmentParameters* cloneFromSelected(const AlgebraicVector& par,
    const AlgebraicSymMatrix& cov) const =0;
 
  /** get alignment parameter selector vector */
  const vector<bool>& selector(void) const;

  /** get number of selected parameters */
  int numSelected(void) const;

  /** get selected parameters */
  AlgebraicVector selectedParameters(void) const;

  /** get selected covariance matrix */
  AlgebraicSymMatrix selectedCovariance(void) const;

  /** get alignment parameters */
  const AlgebraicVector& parameters(void) const;

  /** get parameter covariance matrix */
  const AlgebraicSymMatrix& covariance(void) const;

  /** get (selected) derivatives */
  virtual AlgebraicMatrix derivatives(const TrajectoryStateOnSurface tsos,
    AlignableDet* alidet) const = 0;
  virtual AlgebraicMatrix selectedDerivatives(
    const TrajectoryStateOnSurface tsos, AlignableDet* alidet) const = 0;

  /** set/get pointer to user variables */
  void setUserVariables(AlignmentUserVariables* auv);
  AlignmentUserVariables* userVariables(void) const;

  /** get pointer to alignable (to be removed) */
  Alignable* alignable(void) const;

  /** get number of parameters */
  int size(void) const;

  /** get/set validity flag */
  bool isValid(void) const;
  void setValid(bool v);

  protected:

  /** private helper methods */
  AlgebraicSymMatrix collapseSymMatrix(const AlgebraicSymMatrix& m, 
    const vector<bool>& sel) const; 
  AlgebraicVector collapseVector(const AlgebraicVector& m, 
    const vector<bool>& sel) const;
  AlgebraicSymMatrix expandSymMatrix(const AlgebraicSymMatrix& m, 
    const vector<bool>& sel) const;
  AlgebraicVector expandVector(const AlgebraicVector& m, 
    const vector<bool>& sel) const;

  // data members

  /** pointer to alignable (to be removed) */
  Alignable* theAlignable;

  /** alignment parameters */
  AlgebraicVector theParameters;

  /** alignment parameter covariance matrix */
  AlgebraicSymMatrix theCovariance;

  /** pointer to alignment user variables */
  AlignmentUserVariables* theUserVariables;

  /** defines which parameters are used */
  vector<bool> theSelector;

  /** flag, true if parameters are valid */
  bool bValid;


};

#endif


