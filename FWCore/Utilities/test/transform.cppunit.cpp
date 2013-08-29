/*----------------------------------------------------------------------

Test program for edm::vector_transform class.

 ----------------------------------------------------------------------*/

#include <cassert>
#include <iostream>
#include <string>
#include <boost/algorithm/string.hpp>
#include <cppunit/extensions/HelperMacros.h>

#include "FWCore/Utilities/interface/transform.h"


class testTransform: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testTransform);

  CPPUNIT_TEST(valueTest);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}

  void valueTest();
};

// register the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testTransform);


std::string byvalue_toupper(std::string const & value)
{
  return boost::to_upper_copy( value );
}

const std::string byconstvalue_toupper(std::string const & value)
{
  return boost::to_upper_copy( value );
}

std::string & byref_toupper(std::string const & value)
{
  return * new std::string(boost::to_upper_copy( value ));
}

std::string & byconstref_toupper(std::string const & value)
{
  return * new std::string(boost::to_upper_copy( value ));
}

void testTransform::valueTest()
{
  const std::vector<std::string> input { "Hello", "World" };
  const std::vector<std::string> upper { "HELLO", "WORLD" };
  const std::vector<std::string::size_type> size  { 5, 5 };

  auto test_lambda = edm::vector_transform( input, [](std::string const & value) { return value.size(); } );
  CPPUNIT_ASSERT( size  == test_lambda );

  CPPUNIT_ASSERT( upper == edm::vector_transform(input, byvalue_toupper) );
  CPPUNIT_ASSERT( upper == edm::vector_transform(input, byconstvalue_toupper) );
  CPPUNIT_ASSERT( upper == edm::vector_transform(input, byref_toupper) );
  CPPUNIT_ASSERT( upper == edm::vector_transform(input, byconstref_toupper) );
}
