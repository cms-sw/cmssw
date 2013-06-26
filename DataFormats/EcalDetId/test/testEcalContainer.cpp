/**
   \file
   Test suit for EcalContainer

   \version $Id: testEcalContainer.cpp,v 1.4 2008/10/15 15:14:15 heltsley Exp $

   \note This test is not exaustive     
*/

static const char CVSId[] = "$Id: testEcalContainer.cpp,v 1.4 2008/10/15 15:14:15 heltsley Exp $";

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EcalContainer.h>
#include "FWCore/Utilities/interface/Exception.h"

#include<vector>

#include <iostream>

class testEcalContainer: public CppUnit::TestFixture {

  CPPUNIT_TEST_SUITE(testEcalContainer);
  CPPUNIT_TEST(testContainer);
  CPPUNIT_TEST_SUITE_END();

public:

  void testContainer();
 
}; 

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEcalContainer);

void testEcalContainer::testContainer() {

  EcalContainer<EBDetId, float> ec;
  std::cout << "EcalContainer istantiated." << std::endl;

  for (int ieta=EBDetId::MIN_IETA;ieta<=EBDetId::MAX_IETA;ieta++)
    for (int iphi=EBDetId::MIN_IPHI;iphi<=EBDetId::MAX_IPHI;iphi++)
      {
	//EBDetId Zside 1 
	try
	  {
	    {
	      EBDetId aPositiveId(ieta,iphi);
              ec[aPositiveId] = aPositiveId.hashedIndex();
              EcalContainer<EBDetId, float>::const_iterator citer = ec.find(aPositiveId.rawId());
              if ( citer != ec.end() ) {
                      CPPUNIT_ASSERT( *(ec.find(aPositiveId.rawId())) == aPositiveId.hashedIndex());
              }
	    }

	    //EBDetId Zside -1 
	    {
	      EBDetId aNegativeId(-1*ieta,iphi);
              ec[aNegativeId] = aNegativeId.hashedIndex();
              EcalContainer<EBDetId, float>::const_iterator citer = ec.find(aNegativeId.rawId());
              if ( citer != ec.end() ) {
                      CPPUNIT_ASSERT( *(ec.find(aNegativeId.rawId())) == aNegativeId.hashedIndex());
              }
	    }
	  }
	catch ( cms::Exception &e ) 
	  { 
	    bool cmsExceptionCought=false;
	    CPPUNIT_ASSERT(cmsExceptionCought);
	  }
	catch ( std::exception &e ) 
	  { 
	    bool stdExceptionCought=false;
	    CPPUNIT_ASSERT(stdExceptionCought);
	  }
      }
  std::cout << "Loop finished." << std::endl;
  
  for (int i=0; i!= EBDetId::kSizeForDenseIndexing;++i) {
	    CPPUNIT_ASSERT( ec.item(i) == i);
          //std::cout << ec.item(i) << " " << i << std::endl;
  }

  float i = 0.;
  for ( EcalContainer<EBDetId, float>::const_iterator citer = ec.begin(); citer != ec.end(); citer++) {
	    CPPUNIT_ASSERT( (*citer) == i);
            ++i;
  }
  
  
}
