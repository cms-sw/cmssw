/*
 * $Id: ps_t.cppunit.cc,v 1.22 2009/03/10 04:27:40 rpw Exp $
 */

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <assert.h>

#include <cppunit/extensions/HelperMacros.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"


class testps: public CppUnit::TestFixture
{
  //CPPUNIT_TEST_EXCEPTION(emptyTest,edm::Exception);
  CPPUNIT_TEST_SUITE(testps);
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
  CPPUNIT_TEST(nameAccessTest);
  CPPUNIT_TEST(fileInPathTest);
  CPPUNIT_TEST(testEmbeddedPSet);
  CPPUNIT_TEST(testRegistration);
  CPPUNIT_TEST(testCopyFrom);
  CPPUNIT_TEST(testGetParameterAsString);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp(){}
  void tearDown(){}

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
  void nameAccessTest();
  void fileInPathTest();
  void testEmbeddedPSet();
  void testRegistration();
  void testCopyFrom();
  void testGetParameterAsString();
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
void trackedTestbody(T value)
{
  try {
    edm::ParameterSet p1;
    p1.template addParameter<T>("x", value);
    p1.registerIt();
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

template <class T>
void
untrackedTestbody(T value)
{
  edm::ParameterSet p;
  p.template addUntrackedParameter<T>("x", value);
  CPPUNIT_ASSERT(p.template getUntrackedParameter<T>("x") == value);

  // TODO: When CPPUNIT 1.10.2 arrives, uncomment the following code block.

  //-------------------------------------------------------------------------
  // CPPUNIT_ASSERT_THROW(p.template getUntrackedParameter<T>("does not exist"), 
  // 		       cms::Exception);
  //-------------------------------------------------------------------------

  //-------------------------------------------------------------------------
  // TODO: When CPPUNIT 1.10.2 arrvies, remove this code block.
  try
    {
      // The next line should throw edm::Exception
      p.template getUntrackedParameter<T>("does not exist");
      // We can't use CPPUNIT_ASSERT, because it throws, and that
      // makes it impossible to check for the right exception below.
      assert (0 == "failed to throw a required exception");
     }
   catch (cms::Exception& x)
     {
       // ok, this is expected
     }
   catch (...)
     {
       // Failure!

      // Don't want to use CPPUNIT_ASSERT here, because it throws, and
      // that makes understanding the handling of this catch block too
      // hard for passers-by.
       assert (0 == "threw the wrong kind of exception");
     }
  //-------------------------------------------------------------------------
}

template <class T>
void
testbody(T value)
{
  trackedTestbody<T>(value);
  untrackedTestbody<T>(value);
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
  std::vector<std::string> vs;
  vs.push_back("");
  vs.push_back("1");
  vs.push_back("");
  vs.push_back("three");
  edm::ParameterSet p1;
  p1.addParameter<std::vector<std::string> >("vs",vs);
  p1.registerIt();
  std::vector<std::string> vs2 = p1.getParameter<std::vector<std::string> >("vs");
  //FIXME doesn't count spaces
}


void testps::fileInPathTest()
{
  edm::ParameterSet p;
  edm::FileInPath fip("FWCore/ParameterSet/python/Config.py");
  p.addParameter<edm::FileInPath>("fip", fip);
  CPPUNIT_ASSERT(p.existsAs<edm::FileInPath>("fip"));
  CPPUNIT_ASSERT(p.getParameterNamesForType<edm::FileInPath>()[0] == "fip");
}


void testps::doubleEqualityTest()
{
  edm::ParameterSet p1, p2, p3;
  p1.addParameter<double>("x", 0.1);
  p2.addParameter<double>("x", 1.0e-1);  
  p3.addParameter<double>("x", 0.100);
  p1.registerIt();
  p2.registerIt();
  p3.registerIt();
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
  a1.registerIt();
  a2.registerIt();
  // Negative and positive zero should be coded differently.
  CPPUNIT_ASSERT(a1.toString() != a2.toString());
  CPPUNIT_ASSERT(a1 != a2);
  // Negative and positive zero should test equal.
  CPPUNIT_ASSERT(a1.getParameter<double>("x") == a2.getParameter<double>("x"));
}

void testps::idTest()
{
  edm::ParameterSet a;
  a.registerIt();
  edm::ParameterSetID a_id = a.id();
  edm::ParameterSet b;
  b.addParameter<int>("x", -23);
  b.registerIt();
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
  a.registerIt();
  CPPUNIT_ASSERT( a.exists("pi") );
  CPPUNIT_ASSERT( !a.exists("pie") );

  edm::ParameterSet b;
  b.addParameter<bool>("b", false);
  b.addParameter<std::vector<int> >("three_zeros", std::vector<int>(3,0));
  b.registerIt();

  edm::ParameterSet c;
  c.registerIt();

  edm::ParameterSet d;
  d.addParameter<unsigned int>("hundred", 100);
  d.addParameter<std::vector<double> >("empty", std::vector<double>());
  d.registerIt();

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
  CPPUNIT_ASSERT(psets.size() == 4);
  CPPUNIT_ASSERT(psets[id_a] == a);
  CPPUNIT_ASSERT(psets[id_b] == b);
  CPPUNIT_ASSERT(psets[id_c] == c);
  CPPUNIT_ASSERT(psets[id_d] == d);
}

template <class T>
void 
test_for_name()
{
  edm::ParameterSet preal;
  edm::ParameterSet const& ps = preal;
  // Use 'ps' to make sure we're only getting 'const' access; 
  // use 'preal' when we need to modify the underlying ParameterSet.

  std::vector<std::string> names = ps.getParameterNames();
  CPPUNIT_ASSERT( names.empty() );


  // The following causes failure, because of an apparent GCC bug in comparing bools!
  //T value;
  // Instead, we use this more verbose initialization...
  T value = T();
  preal.template addParameter<T>("x", value);
  names = ps.getParameterNames();
  CPPUNIT_ASSERT( names.size() == 1 );
  CPPUNIT_ASSERT( names[0] == "x" );
  T stored_value = ps.template getParameter<T>(names[0]);
  CPPUNIT_ASSERT( stored_value == value );

  preal.template addUntrackedParameter<T>("y", value);
  preal.registerIt();
  names = ps.getParameterNames();
  CPPUNIT_ASSERT( names.size() == 2 );

  edm::sort_all(names);
  CPPUNIT_ASSERT( edm::binary_search_all(names, "x") );
  CPPUNIT_ASSERT( edm::binary_search_all(names, "y") );
  names = ps.template getParameterNamesForType<T>();
  CPPUNIT_ASSERT( names.size() == 1 );
  edm::sort_all(names);
  CPPUNIT_ASSERT( edm::binary_search_all(names, "x") );
  names = ps.template getParameterNamesForType<T>(false);
  CPPUNIT_ASSERT( names.size() == 1 );
  edm::sort_all(names);
  CPPUNIT_ASSERT( edm::binary_search_all(names, "y") );

  std::string firstString = ps.toString();
  edm::ParameterSet p2(firstString);

  p2.registerIt();
  // equality tests toStringOfTracked internally
  CPPUNIT_ASSERT(ps == p2);
}

void testps::nameAccessTest()
{
  test_for_name<bool>();

  test_for_name<int>();
  test_for_name<std::vector<int> >();

  test_for_name<unsigned int>();
  test_for_name<std::vector<unsigned int> >();

  test_for_name<double>();
  test_for_name<std::vector<double> >();

  test_for_name<std::string>();
  test_for_name<std::vector<std::string> >();
  test_for_name<edm::ParameterSet>();
  test_for_name<std::vector<edm::ParameterSet> >();  

  // Can't make default FileInPath objects...

  // Now make sure that if we put in a parameter of type A, we don't
  // see it when we ask for names of type B != A.
  {
    edm::ParameterSet p;
    p.addParameter<double>("a", 2.5);
    p.registerIt();
    const bool tracked = true;
    std::vector<std::string> names = 
      p.getParameterNamesForType<int>(tracked);
    CPPUNIT_ASSERT( names.empty() ); 
  }

}

void testps::testEmbeddedPSet()
{
  edm::ParameterSet ps;
  edm::ParameterSet psEmbedded, psDeeper;
  psEmbedded.addUntrackedParameter<std::string>("p1", "wham");
  psEmbedded.addParameter<std::string>("p2", "bam");
  psDeeper.addParameter<int>("deepest", 6);
  psDeeper.registerIt();
  edm::InputTag it("label", "instance");
  std::vector<edm::InputTag> vit;
  vit.push_back(it);
  psEmbedded.addParameter<edm::InputTag>("it", it);
  psEmbedded.addParameter<std::vector<edm::InputTag> >("vit", vit);
  psEmbedded.addParameter<edm::ParameterSet>("psDeeper", psDeeper);
  psEmbedded.registerIt();
  ps.addParameter<edm::ParameterSet>("psEmbedded", psEmbedded);
  ps.addParameter<double>("topLevel", 1.);
  ps.addUntrackedParameter<boost::uint64_t>("u64", 64);
  ps.registerIt();

  std::string rep = ps.toString();
  edm::ParameterSet defrosted(rep);
  defrosted.registerIt();
  edm::ParameterSet trackedPart(ps.trackedPart());

  CPPUNIT_ASSERT(defrosted == ps);
  CPPUNIT_ASSERT(trackedPart.exists("psEmbedded"));
  CPPUNIT_ASSERT(trackedPart.getParameterSet("psEmbedded").exists("p2"));
  CPPUNIT_ASSERT(!trackedPart.getParameterSet("psEmbedded").exists("p1"));
  CPPUNIT_ASSERT(trackedPart.getParameterSet("psEmbedded").getParameterSet("psDeeper").getParameter<int>("deepest") == 6);
  CPPUNIT_ASSERT(ps.getUntrackedParameter<boost::uint64_t>("u64") == 64);
  CPPUNIT_ASSERT(!trackedPart.exists("u64"));
}

void testps::testRegistration()
{
  edm::ParameterSet ps;
  edm::ParameterSet psEmbedded, psDeeper;
  psEmbedded.addUntrackedParameter<std::string>("p1", "wham");
  psEmbedded.addParameter<std::string>("p2", "bam");
  psDeeper.addParameter<int>("deepest", 6);
  psDeeper.registerIt();
  edm::InputTag it("label", "instance");
  std::vector<edm::InputTag> vit;
  vit.push_back(it);
  psEmbedded.addParameter<edm::InputTag>("it", it);
  psEmbedded.addParameter<std::vector<edm::InputTag> >("vit", vit);
  psEmbedded.addParameter<edm::ParameterSet>("psDeeper", psDeeper);
  psEmbedded.registerIt();
  ps.addParameter<edm::ParameterSet>("psEmbedded", psEmbedded);
  ps.addParameter<double>("topLevel", 1.);
  ps.addUntrackedParameter<boost::uint64_t>("u64", 64);
  ps.registerIt();
  CPPUNIT_ASSERT(ps.isRegistered());
  CPPUNIT_ASSERT(psEmbedded.isRegistered());
  CPPUNIT_ASSERT(psDeeper.isRegistered());
  psEmbedded.addParameter<std::string>("p3", "slam");
  CPPUNIT_ASSERT(ps.isRegistered());
  CPPUNIT_ASSERT(!psEmbedded.isRegistered());
  CPPUNIT_ASSERT(psDeeper.isRegistered());
}

void testps::testCopyFrom()
{
  edm::ParameterSet psOld;
  edm::ParameterSet psNew;
  edm::ParameterSet psInternal;
  std::vector<edm::ParameterSet> vpset;
  vpset.push_back(psInternal);
  psOld.addParameter<int>("i", 5);
  psOld.addParameter<edm::ParameterSet>("ps", psInternal);
  psOld.addParameter<std::vector<edm::ParameterSet> >("vps", vpset);
  psNew.copyFrom(psOld, "i");
  psNew.copyFrom(psOld, "ps");
  psNew.copyFrom(psOld, "vps");
  CPPUNIT_ASSERT(psNew.existsAs<int>("i"));
  CPPUNIT_ASSERT(psNew.existsAs<edm::ParameterSet>("ps"));
  CPPUNIT_ASSERT(psNew.existsAs<std::vector<edm::ParameterSet> >("vps"));
}

void testps::testGetParameterAsString()
{
  edm::ParameterSet ps;
  edm::ParameterSet psInternal;
  std::vector<edm::ParameterSet> vpset;
  vpset.push_back(psInternal);
  ps.addParameter<int>("i", 5);
  ps.addParameter<edm::ParameterSet>("ps", psInternal);
  ps.addParameter<std::vector<edm::ParameterSet> >("vps", vpset);
  ps.registerIt();
  psInternal.registerIt();
  std::string parStr = ps.getParameterAsString("i");
  std::string psetStr = ps.getParameterAsString("ps");
  std::string vpsetStr = ps.getParameterAsString("vps");
  std::string parStr2 = ps.retrieve("i").toString();
  std::string psetStr2 = ps.retrieveParameterSet("ps").toString();
  std::string vpsetStr2 = ps.retrieveVParameterSet("vps").toString();
  CPPUNIT_ASSERT(parStr == parStr2);
  CPPUNIT_ASSERT(psetStr == psetStr2);
  CPPUNIT_ASSERT(vpsetStr == vpsetStr2);
}

