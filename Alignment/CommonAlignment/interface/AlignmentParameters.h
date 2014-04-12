#ifndef Alignment_CommonAlignment_AlignmentParameters_H
#define Alignment_CommonAlignment_AlignmentParameters_H

#include <vector>

#include "Alignment/CommonAlignment/interface/AlignmentParametersData.h"
#include "Alignment/CommonAlignment/interface/AlignmentUserVariables.h"

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/// \class AlignmentParameters 
///
/// Base class for alignment parameters 
///
/// It contains a parameter vector of size N and a covariance matrix of size NxN. 
/// There is a pointer to the Alignable to which the parameters belong. 
/// There is also a pointer to UserVariables. 
/// Parameters can be enabled/disabled using theSelector. 
/// bValid declares if the parameters are 'valid'.  
/// The methods *selected* set/return only the active
/// parameters/derivatives/covariance as subvector/submatrix
/// of reduced size.
///
///  $Date: 2010/10/26 19:50:21 $
///  $Revision: 1.9 $
/// (last update by $Author: flucke $)

// include and not forward declare to ensure automatic conversion from AlignableDet(Unit): 
// NO: include problems... #include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"
class AlignableDetOrUnitPtr;
class Alignable;
class TrajectoryStateOnSurface;
class RecHit;

class AlignmentParameters 
{

public:

  typedef AlignmentParametersData::DataContainer DataContainer;

  /// Default constructor
  AlignmentParameters();

  /// Constructor from given input
  AlignmentParameters(Alignable* object, const AlgebraicVector& par, 
                      const AlgebraicSymMatrix& cov);

  /// Constructor including selection of active parameters
  AlignmentParameters(Alignable* object, const AlgebraicVector& par, 
                      const AlgebraicSymMatrix& cov, const std::vector<bool>& sel);

  /// Constructor
  AlignmentParameters(Alignable* object, const DataContainer& data );

  /// Destructor
  virtual ~AlignmentParameters();

  /// apply parameters to alignable
  virtual void apply() = 0;
  /// tell type (AlignmentParametersFactory::ParametersType - but no circular dependency)
  virtual int type() const = 0;

  /// Enforce clone methods in derived classes
  virtual AlignmentParameters* clone(const AlgebraicVector& par,
				     const AlgebraicSymMatrix& cov) const = 0;
  virtual AlignmentParameters* cloneFromSelected(const AlgebraicVector& par,
						 const AlgebraicSymMatrix& cov) const = 0;
 
  /// Get alignment parameter selector vector 
  const std::vector<bool>& selector( void ) const;

  /// Get number of selected parameters 
  int numSelected( void ) const;

  /// Get selected parameters
  AlgebraicVector selectedParameters( void ) const;

  /// Get covariance matrix of selected parameters
  AlgebraicSymMatrix selectedCovariance(void) const;

  /// Get alignment parameters
  const AlgebraicVector& parameters(void) const;

  /// Get parameter covariance matrix
  const AlgebraicSymMatrix& covariance(void) const;

  /// Get derivatives of selected parameters
  virtual AlgebraicMatrix derivatives(const TrajectoryStateOnSurface& tsos,
				      const AlignableDetOrUnitPtr &alidet) const = 0;
  virtual AlgebraicMatrix selectedDerivatives( const TrajectoryStateOnSurface& tsos, 
					       const AlignableDetOrUnitPtr &alidet) const;

  /// Set pointer to user variables
  void setUserVariables(AlignmentUserVariables* auv);
  /// Get pointer to user variables
  AlignmentUserVariables* userVariables( void ) const;

  /// Get pointer to corresponding alignable
  Alignable* alignable( void ) const;

  /// How many levels of Alignables with parameters can be found in the 
  /// substructures of the Alignable of these parameters? E.g.
  /// 0: lowest level, i.e. no components of hte Alignable have parameters, 
  /// n: up to n generations of components have parameters (some 'branches' may have less)
  virtual unsigned int hierarchyLevel() const;

  /// Get number of parameters
  int size(void) const;

  /// Get validity flag
  bool isValid(void) const;
  /// Set validity flag
  void setValid(bool v);

protected:

  // private helper methods
  AlgebraicSymMatrix collapseSymMatrix(const AlgebraicSymMatrix& m, 
				       const std::vector<bool>& sel) const; 
  AlgebraicVector collapseVector(const AlgebraicVector& m, 
				 const std::vector<bool>& sel) const;
  AlgebraicSymMatrix expandSymMatrix(const AlgebraicSymMatrix& m, 
				     const std::vector<bool>& sel) const;
  AlgebraicVector expandVector(const AlgebraicVector& m, 
			       const std::vector<bool>& sel) const;

  // data members
  
  Alignable* theAlignable;

  DataContainer theData;

  AlignmentUserVariables* theUserVariables;

  bool bValid; ///< True if parameters are valid


};

#endif


