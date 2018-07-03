#ifndef BXVector_h
#define BXVector_h

// this class is an extension of std::vector
// designed to store objects corresponding to several time-samples (BX)
// the time sample is addressed by an integer index, eg. -1 to 1

#include "DataFormats/Common/interface/FillView.h"
#include "DataFormats/Common/interface/fillPtrVector.h"
#include "DataFormats/Common/interface/setPtr.h"
#include "DataFormats/Common/interface/traits.h"
#include <vector>

template < class T >
class BXVector  {

 public:

  typedef typename std::vector< T >::iterator       iterator;
  typedef typename std::vector< T >::const_iterator const_iterator;
  typedef T value_type;
  typedef typename std::vector< T >::size_type      size_type;

 public:

  // default ctor
  BXVector( unsigned size=0,      // number of objects per BX
	    int bxFirst=0,   // first BX stored
	    int bxLast=0 );  // last BX stored

  // copy ctor
  // BXVector ( const BXVector& vector );

  // dtor
  //~BXVector();

  // assignment operator (pass by value for exception safety)
  //BXVector operator=(BXVector vector );

  // the methods given below are a minimal set
  // other methods from the std::vector interface can be replicated as desired

  // set BX range
  void setBXRange( int bxFirst, int bxLast );

  // set size for a given BX
  void resize( int bx, unsigned size );

  // set size for all BXs
  void resizeAll( unsigned size );
  
  // add one BX to end of BXVector
  void addBX();

  // delete given bunch crossing
  void deleteBX(int bx);

  // get the first BX stored
  int getFirstBX() const;

  // get the last BX stored
  int getLastBX() const;

  // iterator access by BX
  const_iterator begin( int bx ) const;

  // iterator access by BX
  const_iterator end( int bx ) const;

  // get N objects for a given BX
  unsigned size( int bx ) const;

  // get N objects for all BXs together
  unsigned size( ) const { return data_.size();}

  // add element with given BX index
  void push_back( int bx, T object );
 
  // erase element with given location 
  void erase( int bx, unsigned i);
  
  // insert element with given location
  void insert( int bx, unsigned i, T object );

  // clear entire BXVector
  void clear();

  // clear bx
  void clearBX(int bx);

  // access element
  const T& at( int bx, unsigned i ) const;

  // set element
  void set( int bx, unsigned i , const T & object);

  // check if data has empty location
  bool isEmpty(int bx) const;

  // support looping over entire collection (note also that begin() is needed by edm::Ref)  
  const_iterator begin() const {return data_.begin(); }
  const_iterator end() const {return data_.end(); }
  //int bx(const_iterator & iter) const; (potentially useful)
  unsigned int key(const_iterator & iter) const { return iter - begin(); }

  // array subscript operator (incited by TriggerSummaryProducerAOD::fillTriggerObject...)
  T& operator[](std::size_t i) { return data_[i]; }
  const T& operator[](std::size_t i) const { return data_[i]; }

  // edm::View support
  void fillView(edm::ProductID const& id,
		std::vector<void const*>& pointers,
		edm::FillViewHelperVector& helpers) const;
  // edm::Ptr support
  void setPtr(std::type_info const& toType,
	      unsigned long index,
	      void const*& ptr) const;
  void fillPtrVector(std::type_info const& toType,
		     std::vector<unsigned long> const& indices,
		     std::vector<void const*>& ptrs) const;

 private:

  // this method converts integer BX index into an unsigned index
  // used by the internal data representation
  unsigned indexFromBX(int bx) const;
  unsigned numBX() const {return 1 + static_cast<const unsigned>(bxLast_ - bxFirst_); }

 private:

  //  need to keep a record of the BX ranges
  // in order to convert from int BX to the unsigned index
  int bxFirst_;
  int bxLast_;

  /// internal data representation:
  // a flat vector is preferable from the persistency point of view
  // but handling the start/end points for each BX is more complex
  // a second vector is needed to store pointers into the first one
  std::vector< T > data_;
  std::vector<unsigned> itrs_;
};

#include "BXVector.impl"

#endif
