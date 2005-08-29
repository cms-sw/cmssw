/*
 * $Id: ps_t.cppunit.cc,v 1.1 2005/08/19 13:39:04 paterno Exp $
 */

#include <iostream>
#include <limits>
#include <string>

#include <cppunit/extensions/HelperMacros.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"


class testps: public CppUnit::TestFixture
{
  //CPPUNIT_TEST_EXCEPTION(emptyTest,edm::Exception);
  CPPUNIT_TEST_SUITE(testps);
  CPPUNIT_TEST(untrackedTest);
  CPPUNIT_TEST(emptyTest);
  CPPUNIT_TEST(boolTest);
  CPPUNIT_TEST(intTest);
  CPPUNIT_TEST(uintTest);
  CPPUNIT_TEST(doubleTest);
  CPPUNIT_TEST(stringTest);
  CPPUNIT_TEST(doubleEqualityTest);
  CPPUNIT_TEST(negativeZeroTest);
  CPPUNIT_TEST(idTest);
  CPPUNIT_TEST(mapByIdTest);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp(){}
  void tearDown(){}

  void untrackedTest();
  void emptyTest();
  void boolTest();
  void intTest();
  void uintTest();
  void doubleTest();
  void stringTest();
  void doubleEqualityTest();
  void negativeZeroTest();
  void idTest();
  void mapByIdTest();

  // Still more to do...
private:
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testps);

void testps::emptyTest()
{
  edm::ParameterSet p1;
  std::string p1_encoded = p1.toString();
  edm::ParameterSet p2(p1_encoded);
  CPPUNIT_ASSERT (p1 == p2);
}

template <class T>
void testbody(T value)
{
  try {
    edm::ParameterSet p1;
    p1.template addParameter<T>("x", value);
    CPPUNIT_ASSERT(p1.template getParameter<T>("x") == value);
    std::string p1_encoded = p1.toString();
    edm::ParameterSet p2(p1_encoded);
    CPPUNIT_ASSERT(p1 == p2);
    CPPUNIT_ASSERT(p2.template getParameter<T>("x") == value);
  }
  catch (cms::Exception& x)
    {
      std::cerr << "cms Exception caught, message follows\n"
		<< x.what();
      throw;
    }
  catch (std::exception& x)
    {
      std::cerr << "std exception caught, message follows\n"
		<< x.what();
      throw;
    }
  catch (...)
    {
      std::cerr << "Unrecognized exception type thrown\n"
		<< "no details available\n";
      throw;	
    }
}

void testps::untrackedTest()
{
  edm::ParameterSet p1;
  p1.addUntrackedParameter<bool>("x", false);
  CPPUNIT_ASSERT ( p1.getUntrackedParameter<bool>("x") == false );
  try
    {
      // The next line should throw edm::Exception
      p1.getUntrackedParameter<bool>("does not exist");
      CPPUNIT_ASSERT ( "failed to throw a required exception" );
    }
  catch (cms::Exception& x)
    {
      // OK, this is expected
    }
  catch ( ... )
    {
      // Failure!
      CPPUNIT_ASSERT( "threw the wrong kind of exception" );
    }
}

void testps::boolTest()
{
  testbody<bool>(false);
  testbody<bool>(true);
}

void testps::intTest()
{
  testbody<int>(-std::numeric_limits<int>::max());
  testbody<int>(-2112);
  testbody<int>(-0);
  testbody<int>(0);
  testbody<int>(35621);
  testbody<int>(std::numeric_limits<int>::max());
}

void testps::uintTest()
{
  testbody<unsigned int>(0);
  testbody<unsigned int>(35621);
  testbody<unsigned int>(std::numeric_limits<unsigned int>::max());
}

void testps::doubleTest()
{
  testbody<double>(-1.25);
  testbody<double>(-0.0);
  testbody<double>(0.0);
  testbody<double>(1.25);
  //testbody<double>(1.0/0.0);  // parameter set does not handle infinity?
  //testbody<double>(0.0/0.0);  // parameter set does not handle NaN?
  testbody<double>(-2.3456789e-231);
  testbody<double>(-2.3456789e231);
  testbody<double>(2.3456789e-231);
  testbody<double>(2.3456789e231);
  double oneThird = 1.0/3.0;
  testbody<double>(oneThird);
}

void testps::stringTest()
{
  testbody<std::string>("");
  testbody<std::string>("Hello there");
  testbody<std::string>("123");
  testbody<std::string>("This\nis\tsilly\n");  
}


void testps::doubleEqualityTest()
{
  edm::ParameterSet p1, p2, p3;
  p1.addParameter<double>("x", 0.1);
  p2.addParameter<double>("x", 1.0e-1);  
  p3.addParameter<double>("x", 0.100);
  CPPUNIT_ASSERT(p1 == p2);
  CPPUNIT_ASSERT(p1 == p3);
  CPPUNIT_ASSERT(p2 == p3);

  CPPUNIT_ASSERT(p1.toString() == p2.toString());
  CPPUNIT_ASSERT(p1.toString()  == p3.toString());
  CPPUNIT_ASSERT(p2.toString() == p3.toString());
}

void testps::negativeZeroTest()
{
  edm::ParameterSet a1, a2;
  a1.addParameter<double>("x", 0.0);
  a2.addParameter<double>("x", -0.0);
  // Negative and positive zero should be coded differently.
  CPPUNIT_ASSERT(a1.toString() != a2.toString());
  CPPUNIT_ASSERT(a1 != a2);
  // Negative and positive zero should test equal.
  CPPUNIT_ASSERT(a1.getParameter<double>("x") == a2.getParameter<double>("x"));
}

void testps::idTest()
{
  edm::ParameterSet a;
  edm::ParameterSetID a_id = a.id();
  edm::ParameterSet b;
  b.addParameter<int>("x", -23);
  edm::ParameterSetID b_id = b.id();

  CPPUNIT_ASSERT (a != b); 
  CPPUNIT_ASSERT (a.id() != b.id());
}

void testps::mapByIdTest()
{
  // makes parameter sets and ids
  edm::ParameterSet a;
  a.addParameter<double>("pi",3.14);
  a.addParameter<std::string>("name", "Bub");

  edm::ParameterSet b;
  b.addParameter<bool>("b", false);
  b.addParameter<std::vector<int> >("three_zeros", std::vector<int>(3,0));

  edm::ParameterSet c;

  edm::ParameterSet d;
  d.addParameter<unsigned int>("hundred", 100);
  d.addParameter<std::vector<double> >("empty", std::vector<double>());

  edm::ParameterSetID id_a = a.id();
  edm::ParameterSetID id_b = b.id();
  edm::ParameterSetID id_c = c.id();
  edm::ParameterSetID id_d = d.id();

  // fill map
  typedef std::map<edm::ParameterSetID,edm::ParameterSet> map_t;
  map_t   psets;

  psets.insert(std::make_pair(id_a, a));
  psets.insert(std::make_pair(id_b, b));
  psets.insert(std::make_pair(id_c, c));  
  psets.insert(std::make_pair(id_d, d));

  // query map
  CPPUNIT_ASSERT( psets.size() == 4 );
  CPPUNIT_ASSERT( psets[id_a] == a );
  CPPUNIT_ASSERT( psets[id_b] == b );
  CPPUNIT_ASSERT( psets[id_c] == c );
  CPPUNIT_ASSERT( psets[id_d] == d );
}

