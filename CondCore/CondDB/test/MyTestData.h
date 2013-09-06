#ifndef mytestdata_H
#define mytestdata_H

#include <iostream>
#include <string>

#include <boost/serialization/access.hpp> 
#include <boost/serialization/string.hpp> 
#include <boost/serialization/vector.hpp> 
#include <boost/serialization/map.hpp> 

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
private:
  int a;
  float b;
  std::string s;

  // class/struct : RunInfo
  friend class boost::serialization::access;
 private:
   // When the class Archive corresponds to an output archive, the
   // & operator is defined similar to <<.  Likewise, when the class Archive
   // is a type of input archive the & operator is defined similar to >>.
   template<class Archive>
   void serialize(Archive & ar, const unsigned int version) {   ////
     ar & BOOST_SERIALIZATION_NVP(a);
     ar & BOOST_SERIALIZATION_NVP(b);
     ar & BOOST_SERIALIZATION_NVP(s);
   }

};
#endif
