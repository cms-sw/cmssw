/**
   \file
   test file for FEDRawData library

   \author Stefano ARGIRO
   \date 28 Jun 2005
*/

static const char CVSId[] = "$Id: FEDRawData_t.cpp,v 1.2 2006/03/27 10:08:21 argiro Exp $";

#include <cppunit/extensions/HelperMacros.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>

#include <iostream>

class testFEDRawData: public CppUnit::TestFixture {

  CPPUNIT_TEST_SUITE(testFEDRawData);

  CPPUNIT_TEST(testCtor);
  CPPUNIT_TEST(testdata);
 
  CPPUNIT_TEST_SUITE_END();

public:


  void setUp(){}
  void tearDown(){}  
  void testCtor();
  void testdata(); 
 
}; 

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testFEDRawData);


void testFEDRawData::testCtor(){
  FEDRawData f;
  CPPUNIT_ASSERT(f.size()==0);

  FEDRawData f2(24);
  CPPUNIT_ASSERT(f2.size()==size_t(24));
}

void testFEDRawData::testdata(){
  FEDRawData f(48);
  f.data()[0]='a';
  f.data()[1]='b';
  f.data()[47]='c';
 

  const unsigned char * buf= f.data();
  
  CPPUNIT_ASSERT(buf[0] == 'a');
  CPPUNIT_ASSERT(buf[1] == 'b');
  CPPUNIT_ASSERT(buf[47] == 'c');
}


#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
