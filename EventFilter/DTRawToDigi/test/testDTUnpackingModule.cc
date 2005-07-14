/**
   \file
   Test suit for DTUnpackingModule

   \author Stefano ARGIRO
   \version $Id: testDTUnpackingModule.cc,v 1.2 2005/07/13 16:13:12 argiro Exp $
   \date 29 Jun 2005

   \note these tests are not testing anything but the thing not crashing
        
*/

static const char CVSId[] = "$Id: testDTUnpackingModule.cc,v 1.2 2005/07/13 16:13:12 argiro Exp $";

#include <cppunit/extensions/HelperMacros.h>
#include <FWCore/Framework/interface/EventProcessor.h>


class testDTUnpackingModule: public CppUnit::TestFixture {

  CPPUNIT_TEST_SUITE(testDTUnpackingModule);

  CPPUNIT_TEST(testUnpacker);
  CPPUNIT_TEST(testPoolIO);

  CPPUNIT_TEST_SUITE_END();

public:


  void setUp(){}
  void tearDown(){}  
  void testUnpacker();
  void writeOut();
  void testPoolIO();
 
}; 

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testDTUnpackingModule);


void testDTUnpackingModule::testUnpacker(){

  const std::string config=
    "process TEST = { \n"
     "module dtunpacker = DTUnpackingModule{ }\n"
     "module hit = DummyHitFinderModule{ }\n"
     "path p = {dtunpacker, hit}\n"
     "source = DAQFileInputService{ string fileName = \"dtraw.raw\"} \n"
    "}\n";
  
  
  edm::EventProcessor proc(config);
  proc.run();
}

void testDTUnpackingModule::writeOut(){

 const std::string config=
    "process TEST = { \n"
     "module dtunpacker = DTUnpackingModule{ }\n"
     "module out = PoolOutputModule {\n"
     "                   untracked string fileName = 'dtdigis.root'}\n"
     "path p = {dtunpacker, out}\n" 
     "source = DAQFileInputService{ string fileName = \"dtraw.raw\"} \n"
    "}\n";

   edm::EventProcessor proc(config);
   proc.run();   
}

void testDTUnpackingModule::testPoolIO(){

  writeOut();

   const std::string config=
    "process TEST = { \n"
    " module hit = DummyHitFinderModule{ }\n"
    " path p = {hit}\n"
    " source = PoolInputService{ string fileName = \"dtdigis.root\"} \n"
    "}\n";
  

  edm::EventProcessor proc(config);
  proc.run();
}
