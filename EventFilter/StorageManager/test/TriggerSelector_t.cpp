// $Id: TriggerSelector_t.cpp,v 1.3 2012/03/28 13:05:34 mommsen Exp $

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "DataFormats/Common/interface/HLTPathStatus.h"
#include "EventFilter/StorageManager/interface/TriggerSelector.h"

#include "boost/shared_ptr.hpp"

#include <string>
#include <vector>


class testTriggerSelector : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testTriggerSelector);
  CPPUNIT_TEST(test_selector);
  CPPUNIT_TEST_SUITE_END();
  
public:
  void test_selector();
};

void testTriggerSelector::test_selector()
{
  std::vector<std::string> fl;
  fl.push_back( "DiMuon" );
  fl.push_back( "CalibPath" );
  fl.push_back( "DiElectron" );
  fl.push_back( "HighPT" );

  std::string f2 = "(* || 2) || (DixMuo* && (!Di* || CalibPath))";

  std::vector<std::string> triggerList;
  triggerList.push_back("DiMuon");
  triggerList.push_back("CalibPath");
  triggerList.push_back("DiElectron");
  triggerList.push_back("HighPT");

  boost::shared_ptr<stor::TriggerSelector> triggerSelector;

  triggerSelector.reset(new stor::TriggerSelector(f2,triggerList));

  edm::HLTGlobalStatus tr(4);
 
  edm::HLTPathStatus pass = edm::hlt::Pass;
  edm::HLTPathStatus fail = edm::hlt::Fail;

  tr[0] = pass;
  tr[1] = pass;
  tr[2] = fail;
  tr[3] = fail;

  CPPUNIT_ASSERT( triggerSelector->returnStatus(tr) );
}

// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testTriggerSelector);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
