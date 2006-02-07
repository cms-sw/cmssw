/*
 *  $Id: DetSetVector_t.cpp,v 1.1 2006/01/27 21:20:05 paterno Exp $
 *  CMSSW
 *
 */

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <ostream>
#include <vector>

#include "FWCore/EDProduct/interface/traits.h"
#include "FWCore/EDProduct/interface/DetSetVector.h"

using namespace edm;

//------------------------------------------------------
// This is a sample VALUE class, almost the simplest possible.
//------------------------------------------------------

class Value
{
 public:
  // VALUES must be default constructible
  Value() : d_(0.0) { }

  // This constructor is used for testing; it is not required by the
  // concept VALUE.
  explicit Value(double d) : d_(d) { }

  // The compiler-generated copy c'tor seems to do the wrong thing!
  Value(Value const& other) : d_(other.d_) { }

  // This access function is used for testing; it is not required by
  // the concept VALUE.
  double val() const { return d_; }

  // VALUES must be destructible 
  ~Value() {}

  
  // VALUES must be LessThanComparable
  bool operator< (Value const& other) const 
  { return d_ < other.d_; }  

  // The private stuff below is all implementation detail, and not
  // required by the concept VALUE.
 private:
  double        d_;
};


//------------------------------------------------------ 
// The stream insertion operator is not required; it is used here 
// for diagnostic output.
//
// It is inline to avoid multiple definition problems. Note that this
// cc file is *not* a compilation unit; it is actually an include
// file, which the build system combines with others to create a
// compilation unit.
//
//------------------------------------------------------

inline
std::ostream&
operator<< (std::ostream& os, const Value& v)
{
  os << " val: " << v.val();
  return os;
}

typedef edm::DetSetVector<Value> coll_type;
typedef coll_type::detset        detset;

void check_outer_collection_order(coll_type const& c)
{
  if ( c.size() < 2 ) return;
  coll_type::const_iterator i = c.begin();
  coll_type::const_iterator e = c.end();
  // Invariant: sequence from prev to i is correctly ordered
  coll_type::const_iterator prev(i);
  ++i;
  for ( ; i != e; ++i, ++prev )
    {
      // We don't use CPPUNIT_ASSERT because it gives us grossly
      // insufficient context if a failure occurs.
      assert( prev->id < i->id);
    }
}

void check_inner_collection_order(detset const& d)
{
  if ( d.data.size() < 2 ) return;
  detset::const_iterator i = d.data.begin();
  detset::const_iterator e = d.data.end();
  // Invariant: sequence from prev to i is correctly ordered
  detset::const_iterator prev(i);
  ++i;
  for ( ; i != e; ++i, ++prev )
    {
      // We don't use CPPUNIT_ASSERT because it gives us grossly
      // insufficient context if a failure occurs.
      //
      // We don't check that *prev < *i because they might be equal.
      // We don't want to require an op<= or op==.
      assert ( ! (*i < *prev) );
    }
}

void printDetSet(detset const& ds, std::ostream& os)
{
  os << "size: " << ds.data.size() << '\n'
     << "values: ";
  std::copy(ds.data.begin(), ds.data.end(),
	    std::ostream_iterator<detset::value_type>(os, " "));
}


void sanity_check(coll_type const& c)
{
  check_outer_collection_order(c);
  coll_type::const_iterator i = c.begin();
  coll_type::const_iterator e = c.end();
  for ( ; i!=e; ++i) 
    {
      printDetSet(*i, std::cerr);
      std::cerr << '\n';
      check_inner_collection_order(*i);
    }
}


void detsetTest()
{
  std::cerr << "\nStart DetSetVector_t detsetTest()\n";
  detset d;
  Value v1(1.1);
  Value v2(2.2);
  d.id = edm::det_id_type(3);
  d.data.push_back(v1);
  d.data.push_back(v2);
  std::sort( d.data.begin(), d.data.end() );
  check_inner_collection_order(d);  
  std::cerr << "\nEnd DetSetVector_t detsetTest()\n";
}

void traitsTest()
{
  std::cerr << "\nStart DetSetVector_t traitsTest()\n";
  assert( edm::has_postinsert_trait<int>::value == false );
  assert( edm::has_postinsert_trait<coll_type>::value == true );
  std::cerr << "\nEnd DetSetVector_t traitsTest()\n";  
}


void work()
{
  detsetTest();
  traitsTest();


  coll_type c1;
  c1.post_insert();
  sanity_check(c1);
  assert(c1.size() == 0);
  assert(c1.empty() );

  coll_type c2(c1);
  assert(c2.size() == c1.size());
  sanity_check(c2);
  coll_type c;
  sanity_check(c);
  {
    detset    d;
    Value v1(1.1);
    Value v2(2.2);
    d.id = edm::det_id_type(3);
    d.data.push_back(v1);
    d.data.push_back(v2);
    c.insert(d);
    c.post_insert();
  }
  sanity_check(c);
  assert( c.size() == 1 );
  {
    detset    d;
    Value v1(4.1);
    Value v2(3.2);
    d.id = edm::det_id_type(1);
    d.data.push_back(v1);
    d.data.push_back(v2);
    c.insert(d);
    c.post_insert();
  }
  sanity_check(c);
  assert( c.size() == 2 );

  {
    detset    d;
    Value v1(1.1);
    Value v2(1.2);
    Value v3(2.2);
    d.id = edm::det_id_type(10);
    d.data.push_back(v3);
    d.data.push_back(v2);
    d.data.push_back(v1);
    c.insert(d);
    c.post_insert();
  }
  sanity_check(c);
  assert( c.size() == 3);

  coll_type another;
  c.swap(another);
  assert( c.empty() );
  assert( another.size() == 3 );
  sanity_check(c);
  sanity_check(another);

  c.swap(another);
  assert( c.size() == 3);

  {
    // We should not find anything with ID=11
    coll_type::iterator i = c.find(edm::det_id_type(11));
    assert( i == c.end() );

    coll_type::const_iterator ci = 
      static_cast<coll_type const&>(c).find(edm::det_id_type(11));
    assert( ci == c.end() );
  }
  {
    // We should find  ID=10
    coll_type::iterator i = c.find(edm::det_id_type(10));
    assert( i != c.end() );
    assert( i->id == 10);
    assert( i->data.size() == 3 );
  }

  {
    // We should not find ID=100; op[] should throw.
    try
      {
	coll_type::reference r = c[edm::det_id_type(100)];
	assert( "Failed to throw required exception" == 0 );
	assert( &r == 0 ); // to silence warning of unused r
      }
    catch ( edm::Exception const& x )
      {
	// Test we have the right exception category
	assert( x.categoryCode() == edm::errors::InvalidReference);
      }
    catch ( ... )
      {
	assert( "Failed to throw correct exception type" == 0 );
      }
  }

  {
    // We should not find ID=100; op[] should throw.
    try
      {
	coll_type::const_reference r
	  = static_cast<coll_type const&>(c)[edm::det_id_type(100)];
	assert( "Failed to throw required exception" == 0 );
	assert( &r == 0 ); // to silence warning of unused r
      }
    catch ( edm::Exception const& x )
      {
	// Test we have the right exception category
	assert( x.categoryCode() == edm::errors::InvalidReference);
      }
    catch ( ... )
      {
	assert( "Failed to throw correct exception type" == 0 );
      }
  }
  {
    // We should find id = 3
    coll_type const& rc = c;
    coll_type::const_reference r = rc[3];
    assert( r.id == edm::det_id_type(3) );
    assert( r.data.size() == 2 );

    coll_type::reference r2 = c[3];
    assert( r2.id == edm::det_id_type(3) );
    assert( r2.data.size() == 2 );
  }

  {
    // We should not find id = 17, but a new empty DetSet should be
    // inserted, carrying the correct DetId.
    coll_type::size_type oldsize = c.size();
    coll_type::reference r = c.find_or_insert(edm::det_id_type(17));
    coll_type::size_type newsize = c.size();
    assert( newsize > oldsize );
    assert( newsize = (oldsize+1) );
    assert( r.id == edm::det_id_type(17) );
    assert( r.data.size() == 0 );
    r.data.push_back(Value(10.1));
    r.data.push_back(Value(9.1));
    r.data.push_back(Value(4.0));
    r.data.push_back(Value(4.0));
    c.post_insert();
    sanity_check(c);
  }


//   {
//     // We should find ID=10, and be able to add more to it.
//     coll_type::iterator i = c.find_or_insert(edm::det_id_type(10));
//     assert( i != c.end() );
//     assert( i->id == 10);
//     assert( i->data.size() == 3 );
//     Value v(-1.2);
//     i->push_back(v);
//     // We've now made the vector be out-of-order...
//     c.post_insert(); // and now it is fixed.
//     sanity_check(c);

//     i = c.find(edm::det_id_type(10));
//     assert( i != c.end() );
//     assert( i->id == 10);
//     assert( i->data.size() == 4 );    
//   }
}



int main()
{
  int rc = -1;
  try 
    { 
      work(); 
      rc = 0;
    }
  catch ( edm::Exception const& x )
    {
      std::cerr << "Exception: " << x << '\n';
      rc = 1;
    }
  catch ( ... )
    {
      std::cerr << "Unknown exception\n";
      rc = 2;
    }
  return rc;
}



