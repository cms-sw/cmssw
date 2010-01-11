#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestRunner.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TextTestProgressListener.h>
#include <cppunit/CompilerOutputter.h>

#ifndef TestMuScleFit_cc
#define TestMuScleFit_cc

class Complex
{
public:
  Complex( const double realInput, const double imgInput) :
    real(realInput),
    img(imgInput)
  {
  }
  Complex operator+( const Complex & number )
  {
    Complex temp(*this);
    temp.real += number.real;
    temp.img += number.img;
    return temp;
  }
  bool operator==(const Complex & number) const
  {
    return( real == number.real && img == number.img );
  }

  double real;
  double img;
};

class TestMuScleFit : public CppUnit::TestFixture {
public:
  TestMuScleFit() {}

  void setUp()
  {
    m_10_1 = new Complex(10, 1);
    m_1_1 = new Complex(1, 1);
    m_11_2 = new Complex(11, 2);
  }
  void tearDown()
  {
    delete m_10_1;
    delete m_1_1;
    delete m_11_2;
  }
  void testEquality()
  {
    CPPUNIT_ASSERT( *m_10_1 == *m_10_1 );
    CPPUNIT_ASSERT( !(*m_10_1 == *m_11_2) );
  }
  void testAddition()
  {
    CPPUNIT_ASSERT( *m_10_1 + *m_1_1 == *m_11_2 );
  }

  // Declare and build the test suite
  CPPUNIT_TEST_SUITE( TestMuScleFit );
  CPPUNIT_TEST( testEquality );
  CPPUNIT_TEST( testAddition );
  CPPUNIT_TEST_SUITE_END();

  Complex * m_10_1, * m_1_1, * m_11_2;
};

// Register the test suite in the registry.
// This way we will have to only pass the registry to the runner
// and it will contain all the registered test suites.
CPPUNIT_TEST_SUITE_REGISTRATION( TestMuScleFit );

#endif
