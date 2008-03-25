
#include <iostream>


#include "FWCore/ParameterSet/interface/Makers.h"

#include <cppunit/extensions/HelperMacros.h>

using namespace std;
using namespace edm;

// ----------------------------------------------
class testmaker: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testmaker);
CPPUNIT_TEST(makerTest);
CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}
  void makerTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testmaker);

void testmaker::makerTest()
//int main()
{
  string param1 = 
    "string module_type = \"TestMod\"\n "
    " string module_label = \"t1\"";

  string param2 = 
    "string module_type = \"TestMod\" "
    "string module_label = \"t2\"";
    
  /*try {

    edmplugin::PluginManager::configure(edmplugin::standard::config());
    Factory* f = Factory::get();

    //Factory::Iterator ib(f->begin()),ie(f->end());
    //for(;ib!=ie;++ib)
    //  {
    //cout << (*ib)->name() << endl;
    // }

    boost::shared_ptr<ParameterSet> p1 = makePSet(*edm::pset::parse(param1.c_str()));;
    boost::shared_ptr<ParameterSet> p2 = makePSet(*edm::pset::parse(param2.c_str()));;

    cerr << p1->getParameter<std::string>("@module_type");

    edm::ActionTable table;

    edm::ProductRegistry preg;

    auto_ptr<Worker> w1 = f->makeWorker(*p1, preg, table, "PROD", 0, 0);
    auto_ptr<Worker> w2 = f->makeWorker(*p2, preg, table, "PROD", 0, 0);
  }
  catch(cms::Exception& e) {
      std::cerr << "cms::Exception: " << e.explainSelf() << std::endl;
      throw;
  }
  catch(std::exception& e) {
      std::cerr << "std::Exception: " << e.what() << std::endl;
      throw;
  }
  catch(...) {
      std::cerr << "weird exception" << endl;
      throw;
  }

  return 0;*/
}
