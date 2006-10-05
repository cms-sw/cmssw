#ifndef Alignment_KalmanAlignmentAlgorithm_LookUpTable_h
#define Alignment_KalmanAlignmentAlgorithm_LookUpTable_h

#include <vector>
#include <map>
#include <algorithm>

using namespace std;

/// A look-up table used by the MetricsCalculator. The values stored in this table are identified by
/// two indices, like in a matrix. The indices and the associated value can be of arbitrary type, the
/// indices only have to be comparable.


template< class Ti, class Tv >
class LookupTable {

 public:

  typedef typename map< Ti, map< Ti, Tv > >::iterator iterator;
  typedef typename map< Ti, map< Ti, Tv > >::const_iterator const_iterator;

  /// Get value associated with indices I and J.
  Tv & operator() ( Ti I, Ti J ) { return theTable[I][J]; }

  /// Get all values associated with first index I and arbitrary second index.
  map< Ti, Tv > getRow( Ti I ) { return theTable[I]; }
  ///Associate values to first index I and arbitrary second index.
  void setRow( Ti & I, map< Ti, Tv > & row, bool eraseOldRow = false );

  /// Get all values associated with arbitrary first index and second index J.
  map< Ti, Tv > getColumn( Ti J );
  /// Set all values associated with arbitrary first index and second index J.
  void setColumn( Ti & J, map< Ti, Tv > & col, bool eraseOldColumn = false );

  iterator end( void ) { return theTable.end(); }
  iterator begin( void ) { return theTable.begin(); }

  /// Clear the table.
  void clear( void ) { theTable.clear(); }
  /// Very low-level output (cout).
  void print( void );

 private:

  void eraseOldAndSetNewRow( Ti & I, map< Ti, Tv > & row );
  void setNewRowElements( Ti & I, map< Ti, Tv > & row );

  void eraseOldAndSetNewColumn( Ti & J, map< Ti, Tv > & col );
  void setNewColumnElements( Ti & J, map< Ti, Tv > & col );

  map< Ti, map< Ti, Tv > > theTable;
};

#endif
