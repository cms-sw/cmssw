
#include "FWCore/Utilities/interface/EDMException.h"

#include <iostream>
#include <string>
#include <iomanip>
#include <typeinfo>

//using namespace cms;

void func3()
{
  throw edm::Exception(edm::errors::NotFound)
    << "This is just a test"
    << std::endl;
}

void func2()
{
  func3();
}

void func1()
{
  try {
      func2();
  }
  catch (edm::Exception& e) {
    //std::cerr << "GOT HERE" << std::endl;
    throw edm::Exception(edm::errors::Unknown,"In func2",e)
	<< "Gave up";
  }
  catch (cms::Exception& e) {
    //std::cerr << "GOT HERE2 " << typeid(e).name() << std::endl;
    throw cms::Exception("edm::errors::Unknown","In func2 bad place",e)
	<< "Gave up";
  }
  
}

const char answer[] = 
  "---- Unknown BEGIN\n"
  "In func2\n"
  "---- NotFound BEGIN\n"
  "This is just a test\n" 
  "---- NotFound END\n"
  "Gave up\n"
  "---- Unknown END\n"
  ;

const char* correct[] = { "Unknown","NotFound" };

int main()
{
  try {
    func1();
  }
  catch (cms::Exception& e) {
    std::cerr << "*** main caught Exception, output is ***\n"
	 << "(" << e.explainSelf() << ")"
	 << "*** After exception output ***"
	 << std::endl;

    std::cerr << "\nCategory name list:\n";

#if 1
    if(e.explainSelf() != answer) {
	std::cerr << "not right answer\n(" << answer << ")\n"
	     << std::endl;
	abort();
    }
#endif

    cms::Exception::CategoryList::const_iterator i(e.history().begin()),
	b(e.history().end());

    if(e.history().size() != 2) {
      std::cerr << "Exception history is bad"  << std::endl;
      abort();
    }

    for(int j=0; i != b; ++i, ++j) {
      std::cout << "  " << *i << "\n";
      if(*i != correct[j]) {
        std::cerr << "bad category " << *i << std::endl;
        abort();
      }
    }
  }
  return 0; 
}
