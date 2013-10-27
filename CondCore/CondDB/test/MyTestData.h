#ifndef mytestdata_H
#define mytestdata_H

#include <iostream>
#include <string>

#include "CondFormats/Serialization/interface/Serializable.h"

class MyTestData {
public:

  MyTestData():
    a( 0 ),
    b( 0. ),
    s(""){
  }
  MyTestData( int seed ):
    a( seed ),
    b( seed + 1.1 ),
    s( "Have a nice day!" ){
  }
  void print(){
    std::cout <<"MyTestData: a="<<a<<" b="<<b<<" s="<<s<<std::endl;
  }

  bool operator==( const MyTestData& rhs ) const {
    if( a != rhs.a ) return false;
    if( b != rhs.b ) return false;
    if( s != rhs.s ) return false;
    return true;
  }
  bool operator!=( const MyTestData& rhs ) const {
    return !operator==( rhs );
  }
private:
  int a;
  float b;
  std::string s;

  COND_SERIALIZABLE;
};

#include <boost/serialization/nvp.hpp>

template <class Archive>
void MyTestData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(a);
    ar & BOOST_SERIALIZATION_NVP(b);
    ar & BOOST_SERIALIZATION_NVP(s);
}

#endif
