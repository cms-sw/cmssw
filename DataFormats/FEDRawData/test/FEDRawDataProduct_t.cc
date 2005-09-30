/**
   \file
   unit test file for class FEDRawDataProduct 

   \author Stefano ARGIRO
   \version $Id: FEDRawDataProduct_t.cc,v 1.1 2005/07/06 16:37:54 argiro Exp $
   \date 28 Jun 2005
*/

static const char CVSId[] = "$Id: FEDRawDataProduct_t.cc,v 1.1 2005/07/06 16:37:54 argiro Exp $";


#include <cppunit/extensions/HelperMacros.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

class testFEDRawDataProduct: public CppUnit::TestFixture {

  CPPUNIT_TEST_SUITE(testFEDRawDataProduct);

  CPPUNIT_TEST(testInsertAndReadBack);
 
  CPPUNIT_TEST_SUITE_END();

public:


  void setUp(){}
  void tearDown(){}  
  void testInsertAndReadBack();
}; 

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testFEDRawDataProduct);


void testFEDRawDataProduct::testInsertAndReadBack(){

  raw::FEDRawData f1(3);
  f1.data()[0] = 'a';
  f1.data()[1] = 'b';
 

  raw::FEDRawData f2(10);
  f2.data()[0] = 'd';
  f2.data()[1] = 'e';
 
  raw::FEDRawDataCollection fp;
  fp.FEDData(12) =  f1;
  fp.FEDData(121) = f2;

  CPPUNIT_ASSERT(fp.FEDData(12).data()[0]=='a');
  CPPUNIT_ASSERT(fp.FEDData(12).data()[1]=='b');

  CPPUNIT_ASSERT(fp.FEDData(121).data()[0]=='d');
  CPPUNIT_ASSERT(fp.FEDData(121).data()[1]=='e');
 

}


