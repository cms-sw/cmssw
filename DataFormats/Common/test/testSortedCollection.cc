/*
 *  sortedcollection_t.cppunit.cc
 *  CMSSW
 *
 */

// This is a serious hack! Remove it as soon as we get templated on 'key'.
typedef int DetId;

#include <algorithm>
#include <iostream>
#include <iterator>
#include <ostream>
#include <vector>

#include <catch2/catch_all.hpp>
#include "DataFormats/Common/interface/SortedCollection.h"

using namespace edm;

//------------------------------------------------------
// This is a sample VALUE class, almost the simplest possible.
//------------------------------------------------------

class Value {
public:
  // The following is a hack, to make this class Value look like the
  // real use case.
  typedef int DetId;

  // VALUES must declare a key_type
  typedef DetId key_type;

  // VALUES must be default constructible
  Value() : d_(0.0), id_(0) {}

  // This constructor is used for testing; it is not required by the
  // concept VALUE.
  Value(double d, DetId anId) : d_(d), id_(anId) {}

  // This access function is used for testing; it is not required by
  // the concept VALUE.
  double val() const { return d_; }

  // VALUES must have a const member function id() that returns a
  // key_type. N.B.: here, DetId is key_type.
  DetId id() const { return id_; }

  // VALUES must be destructible
  ~Value() {}

  // The private stuff below is all implementation detail, and not
  // required by the concept VALUE.
private:
  double d_;
  DetId id_;
};

//------------------------------------------------------
// If one wants to test SortedCollections using operator==, then one
// must use a VALUE that is equality comparable. Otherwise, VALUE need
// not be equality comparable.
//------------------------------------------------------

bool operator==(Value const& a, Value const& b) { return (a.id() == b.id()) && (a.val() == b.val()); }

//------------------------------------------------------
// The stream insertion operator is not required; it is used here
// for diagnostic output.
//------------------------------------------------------
std::ostream& operator<<(std::ostream& os, const Value& v) {
  os << "id: " << v.id() << " val: " << v.val();
  return os;
}

typedef edm::SortedCollection<Value, StrictWeakOrdering<Value> > scoll_type;

template <class T, class SORT>
void append_to_both(edm::SortedCollection<T, SORT>& sc, std::vector<T>& vec, const T& t) {
  sc.push_back(t);
  vec.push_back(t);
}

TEST_CASE("SortedCollection", "[SortedCollection]") {
  SECTION("constructTest") {
    scoll_type c1;
    REQUIRE(c1.size() == 0);

    scoll_type c2(20);
    REQUIRE(c2.size() == 20);

    std::vector<Value> values(3);
    scoll_type c3(values);
    REQUIRE(c3.size() == values.size());

    scoll_type c4(c3);
    REQUIRE(c4.size() == c3.size());
    REQUIRE(c3 == c4);
  }

  SECTION("insertTest") {
    scoll_type c;
    Value v1(1.1, 1);
    Value v2(2.2, 2);

    c.push_back(v1);
    REQUIRE(c.size() == 1);
    c.push_back(v2);
    REQUIRE(c.size() == 2);
  }

  SECTION("accessTest") {
    scoll_type c;
    std::vector<Value> vec;
    append_to_both(c, vec, Value(1.5, 3));
    append_to_both(c, vec, Value(2.5, 200));
    append_to_both(c, vec, Value(3.5, 1));
    append_to_both(c, vec, Value(4.5, 1001));
    append_to_both(c, vec, Value(5.5, 2));
    //  append_to_both(c, vec, Value(10.5, 3));

    REQUIRE(c.size() == vec.size());
    c.sort();

    // DO NOT ADD ANY MORE ITEMS TO c!
    REQUIRE(c.size() == vec.size());

    {
      scoll_type::iterator i = c.find(DetId(100));  // does not exist!
      REQUIRE(i == c.end());
    }

    {
      std::cerr << "Dumping SortedCollection" << std::endl;
      std::copy(c.begin(), c.end(), std::ostream_iterator<Value>(std::cerr, "\n"));
    }

    {
      std::vector<Value>::const_iterator i = vec.begin();
      std::vector<Value>::const_iterator e = vec.end();
      std::cerr << "There are " << vec.size() << " searches to do...\n";
      while (i != e) {
        DetId id = i->id();
        std::cerr << "Looking for id: " << id << "...   ";
        //scoll_type::iterator loc = c.find(i->id());
        scoll_type::iterator loc = c.find(id);
        if (loc == c.end())
          std::cerr << "Failed to find this id!\n";
        else
          std::cerr << "Found it, record is: " << *loc << '\n';
        REQUIRE(loc != c.end());
        REQUIRE(*loc == *i);
        ++i;
      }
    }
  }

  SECTION("swapTest") {
    scoll_type c;
    std::vector<Value> vec;
    append_to_both(c, vec, Value(1.5, 3));
    append_to_both(c, vec, Value(2.5, 200));
    append_to_both(c, vec, Value(3.5, 1));
    append_to_both(c, vec, Value(4.5, 1001));
    append_to_both(c, vec, Value(5.5, 2));

    {
      scoll_type copy(c);
      REQUIRE(copy == c);
      scoll_type empty;
      REQUIRE(empty.empty());

      empty.swap(copy);
      REQUIRE(copy.empty());
      REQUIRE(empty == c);
    }

    {
      std::vector<Value> copy(vec);
      scoll_type empty;
      REQUIRE(empty.empty());

      empty.swap_contents(copy);
      REQUIRE(copy.empty());
      REQUIRE(empty == vec);
    }
  }

  SECTION("frontbackTest") {
    scoll_type c;
    std::vector<Value> vec;
    append_to_both(c, vec, Value(1.5, 3));
    append_to_both(c, vec, Value(2.5, 200));
    append_to_both(c, vec, Value(3.5, 1));
    append_to_both(c, vec, Value(4.5, 1001));
    append_to_both(c, vec, Value(5.5, 2));

    c.sort();

    REQUIRE(c.front() == Value(3.5, 1));
    REQUIRE(c.back() == Value(4.5, 1001));

    scoll_type const& cr = c;
    REQUIRE(cr.front() == Value(3.5, 1));
    REQUIRE(cr.back() == Value(4.5, 1001));
  }

  SECTION("squarebracketTest") {
    scoll_type c;
    std::vector<Value> vec;
    append_to_both(c, vec, Value(1.5, 3));
    append_to_both(c, vec, Value(2.5, 200));
    append_to_both(c, vec, Value(3.5, 1));
    append_to_both(c, vec, Value(4.5, 1001));
    append_to_both(c, vec, Value(5.5, 2));

    c.sort();

    REQUIRE(c[0] == Value(3.5, 1));
    REQUIRE(c[1] == Value(5.5, 2));
    REQUIRE(c[2] == Value(1.5, 3));
    REQUIRE(c[3] == Value(2.5, 200));
    REQUIRE(c[4] == Value(4.5, 1001));

    scoll_type const& cr = c;
    REQUIRE(cr[0] == Value(3.5, 1));
    REQUIRE(cr[1] == Value(5.5, 2));
    REQUIRE(cr[2] == Value(1.5, 3));
    REQUIRE(cr[3] == Value(2.5, 200));
    REQUIRE(cr[4] == Value(4.5, 1001));
  }
}
