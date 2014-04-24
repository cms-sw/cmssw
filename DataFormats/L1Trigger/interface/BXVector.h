#ifndef BXVector_h
#define BXVector_h

// this class is an extension of std::vector
// designed to store objects corresponding to several time-samples (BX)
// the time sample is addressed by an integer index, eg. -1 to 1

#include <vector>

template < class T >
class BXVector  {

 public:

  typedef typename std::vector< T >::iterator       iterator;
  typedef typename std::vector< T >::const_iterator const_iterator;

 public:

  // default ctor
  BXVector( int size=0,      // number of objects per BX
	    int bxFirst=0,   // first BX stored
	    int bxLast=0 );  // last BX stored

  // copy ctor
  BXVector ( const BXVector& vector );

  // dtor
  ~BXVector();

  // assignment operator (pass by value for exception safety)
  //BXVector operator=( BXVector vector );

  // the methods given below are a minimal set
  // other methods from the std::vector interface can be replicated as desired

  // set BX range
  void setBXRange( int bxFirst, int bxLast );

  // set size for a given BX
  void resize( int bx, int size );

  // set size for all BXs
  void resizeAll( int size );

  // get the first BX stored
  int getFirstBX() const;

  // get the last BX stored
  int getLastBX() const;

  // get N objects for a given BX
  unsigned size( int bx ) const;

  // add element with given BX index
  void push_back( int bx, T object );

  // add clear member
  void clear();

  // random access
  const T& at( int bx, int i ) const;

  // iterator access by BX
  const_iterator begin( int bx ) const;

  // iterator access by BX
  const_iterator end( int bx ) const;

 private:

  // this method converts integer BX index into an unsigned index
  // used by the internal data representation
  unsigned indexFromBX(int bx) const;


 private:

  //  need to keep a record of the BX ranges
  // in order to convert from int BX to the unsigned index
  int bxFirst_;
  int bxLast_;

  /// internal data representation

  // Version 1
  // this version is easy to handle
  // but nested template containers are disfavoured by persistency layer
  std::vector< std::vector< T > > data_;

  // Version 2
  // a flat vector is preferable from the persistency point of view
  // but handling the start/end points for each BX is more complex
  // a second vector is needed to store pointers into the first one
  /* std::vector< T > data_; */
  /* std::vector< std::vector::iterator > itrs_; */

};

#include "BXVector.impl"

#endif
