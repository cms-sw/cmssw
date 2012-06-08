#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "EventFilter/StorageManager/interface/EnquingPolicyTag.h"

#include <sstream>
#include <string>
#include <vector>

using namespace std;

class testEnquingPolicyTag : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testEnquingPolicyTag);
  CPPUNIT_TEST(valid_tags);
  CPPUNIT_TEST_SUITE_END();
  
public:
  void valid_tags();
};

void testEnquingPolicyTag::valid_tags()
{
  typedef std::vector<std::string> SVec;
  SVec tags;
  tags.push_back( "Discard New" );
  tags.push_back( "Discard Old" );
  tags.push_back( "Fail If Full" );
  tags.push_back( "Max" );
  
  for( unsigned int i = 0; i < tags.size(); ++i )
  {
    ostringstream oss;
    oss << stor::enquing_policy::PolicyTag( i );
    CPPUNIT_ASSERT( oss.str() == tags[i] );
  }
}

// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testEnquingPolicyTag);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
