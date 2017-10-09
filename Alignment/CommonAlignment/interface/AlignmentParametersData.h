#ifndef Alignment_CommonAlignment_AlignmentParametersData_h
#define Alignment_CommonAlignment_AlignmentParametersData_h

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

class AlignmentParametersData : public ReferenceCounted
{

public:

  typedef ReferenceCountingPointer<AlignmentParametersData> DataContainer;

  /// Default constructor.
  AlignmentParametersData( void );

  /// Constructor from parameters vector, covariance matrix and selection vector.
  /// NOTE: The input data must live on the heap and must not be deleted by the user.
  AlignmentParametersData( AlgebraicVector* param,
			   AlgebraicSymMatrix* cov,
			   std::vector<bool>* sel );

  /// Constructor from parameters vector, covariance matrix and selection vector.
  AlignmentParametersData( const AlgebraicVector& param,
			   const AlgebraicSymMatrix& cov,
			   const std::vector<bool>& sel );

  /// Constructor from parameters vector and covariance matrix.
  /// NOTE: The input data must live on the heap and must not be deleted by the user.
  AlignmentParametersData( AlgebraicVector* param,
			   AlgebraicSymMatrix* cov );

  /// Constructor from parameters vector and covariance matrix.
  AlignmentParametersData( const AlgebraicVector& param,
			   const AlgebraicSymMatrix& cov );

  ~AlignmentParametersData( void );

  /// Access to the parameter vector.
  const AlgebraicVector& parameters( void ) const { return *theParameters; }

  /// Access to the covariance matrix
  const AlgebraicSymMatrix& covariance( void ) const { return *theCovariance; } 

  /// Access to the selection vector.
  const std::vector<bool>& selector( void ) const { return *theSelector; }

  /// Access to the number of selected parameters.
  int numSelected( void ) { return theNumSelected; }

  /// Check if the size of the parameters vector, the size of the covariance matrix,
  /// the size of the selector and the number of selected parameters is consistent.
  /// An exception of type "LogicError" is thrown in case of any inconsistencies.
  void checkConsistency( void ) const;

private:

  AlgebraicVector* theParameters;
  AlgebraicSymMatrix* theCovariance;
  std::vector<bool>* theSelector;
  int theNumSelected;
};

#endif
