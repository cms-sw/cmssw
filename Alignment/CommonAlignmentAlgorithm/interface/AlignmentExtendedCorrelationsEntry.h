#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentExtendedCorrelationsEntry_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentExtendedCorrelationsEntry_h

#include <vector>
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/// Data container for a correlations matrix (represented by a vector of floats), with
/// basic access functions. NOTE: This class is designed specifically for the use within
/// AlignmentExtendedCorrelationsStore, and is not intended to be used elsewhere.

class AlignmentExtendedCorrelationsEntry {
public:
  /// Default constructor.
  AlignmentExtendedCorrelationsEntry(void);

  /// Constructor. Leaves the correlations matrix uninitialized.
  explicit AlignmentExtendedCorrelationsEntry(short unsigned int nRows, short unsigned int nCols);

  /// Constructor. Initializes all elements of the correlations matrix to the given value.
  explicit AlignmentExtendedCorrelationsEntry(short unsigned int nRows, short unsigned int nCols, const float init);

  /// Constructor from CLHEP matrix.
  explicit AlignmentExtendedCorrelationsEntry(const AlgebraicMatrix& mat);

  /// Destructor.
  ~AlignmentExtendedCorrelationsEntry(void) {}

  /// Read or write an element of the correlations matrix. NOTE: Indexing starts from [0,0].
  inline float& operator()(short unsigned int iRow, short unsigned int jCol) { return theData[iRow * theNCols + jCol]; }

  /// Read or write an element of the correlations matrix. NOTE: Indexing starts from [0,0].
  inline const float operator()(short unsigned int iRow, short unsigned int jCol) const {
    return theData[iRow * theNCols + jCol];
  }

  /// Get the number of rows of the correlation matrix.
  inline const short unsigned int numRow(void) const { return theNRows; }

  /// Get the number of columns of the correlation matrix.
  inline const short unsigned int numCol(void) const { return theNCols; }

  /// Multiply all elements of the correlations matrix with a given number.
  void operator*=(const float multiply);

  /// Retrieve the correlation matrix in a CLHEP matrix representation;
  AlgebraicMatrix matrix(void) const;

  //   /// Get the counter's value.
  //   inline const int counter( void ) const { return theCounter; }

  //   /// Increase the counter's value by 1.
  //   inline void incrementCounter( void ) { ++theCounter; }

  //   /// Decrease the counter's value by 1.
  //   inline void decrementCounter( void ) { --theCounter; }

private:
  //   int theCounter;

  short unsigned int theNRows;
  short unsigned int theNCols;

  std::vector<float> theData;
};

#endif
