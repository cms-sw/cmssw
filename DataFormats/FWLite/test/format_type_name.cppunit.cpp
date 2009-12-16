/*
 *  format_type_name.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 09/12/09.
 *
 */

#include <cppunit/extensions/HelperMacros.h>
#include <iostream>
#include "DataFormats/FWLite/interface/format_type_name.h"


class testFormatTypeName: public CppUnit::TestFixture
{
   CPPUNIT_TEST_SUITE(testFormatTypeName);
   
   CPPUNIT_TEST(test);
   
   CPPUNIT_TEST_SUITE_END();
public:
      void setUp(){}
   void tearDown(){}
   
   void test();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testFormatTypeName);


void testFormatTypeName::test()
{
  typedef std::pair<std::string,std::string> Values; 
  std::vector<Values> classToFriendly;
  classToFriendly.push_back( Values("Foo","Foo") );
  classToFriendly.push_back( Values("bar::Foo","bar_1Foo") );
  classToFriendly.push_back( Values("std::vector<Foo>","std_1vector_9Foo_0") );
  classToFriendly.push_back( Values("std::vector<bar::Foo>","std_1vector_9bar_1Foo_0") );
  classToFriendly.push_back( Values("V<A,B>","V_9A_3B_0") );
  classToFriendly.push_back( Values("edm::ExtCollection<std::vector<reco::SuperCluster>,reco::SuperClusterRefProds>","edm_1ExtCollection_9std_1vector_9reco_1SuperCluster_0_3reco_1SuperClusterRefProds_0") );
  classToFriendly.push_back( Values("A<B<C>, D<E> >","A_9B_9C_0_3_4D_9E_0_4_0"));
  classToFriendly.push_back( Values("A<B<C<D> > >","A_9B_9C_9D_0_4_0_4_0"));
  classToFriendly.push_back( Values("A<B<C,D>, E<F> >","A_9B_9C_3D_0_3_4E_9F_0_4_0"));
  classToFriendly.push_back( Values("Aa<Bb<Cc>, Dd<Ee> >","Aa_9Bb_9Cc_0_3_4Dd_9Ee_0_4_0"));
  classToFriendly.push_back( Values("Aa<Bb<Cc<Dd> > >","Aa_9Bb_9Cc_9Dd_0_4_0_4_0"));
  classToFriendly.push_back( Values("Aa<Bb<Cc,Dd>, Ee<Ff> >","Aa_9Bb_9Cc_3Dd_0_3_4Ee_9Ff_0_4_0"));
  classToFriendly.push_back( Values("Aa<Bb<Cc,Dd>, Ee<Ff,Gg> >","Aa_9Bb_9Cc_3Dd_0_3_4Ee_9Ff_3Gg_0_4_0"));
                                 
  for(std::vector<Values>::iterator itInfo = classToFriendly.begin(),
      itInfoEnd = classToFriendly.end();
      itInfo != itInfoEnd;
      ++itInfo) {
    //std::cout <<itInfo->first<<std::endl;
    if( itInfo->second != fwlite::format_type_to_mangled(itInfo->first) ) {
      std::cout <<"class name: '"<<itInfo->first<<"' has wrong mangled name \n"
      <<"expect: '"<<itInfo->second<<"' got: '"<<fwlite::format_type_to_mangled(itInfo->first)<<"'"<<std::endl;
      CPPUNIT_ASSERT(0 && "expected mangled name does not match actual mangled name");
    }
    if( itInfo->first != fwlite::unformat_mangled_to_type(itInfo->second) ) {
      std::cout <<"mangled name: '"<<itInfo->second<<"' has wrong type name \n"
      <<"expect: '"<<itInfo->first<<"' got: '"<<fwlite::unformat_mangled_to_type(itInfo->second)<<"'"<<std::endl;
      CPPUNIT_ASSERT(0 && "expected type name does not match actual type name");
    }
    
  }
}

