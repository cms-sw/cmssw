
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <string>
#include <iomanip>

using namespace cms;

struct Thing : public Exception
{
  Thing(const std::string& msg):Exception("Thing",msg) { }
};

std::ostream& operator<<(std::ostream& os, const Thing& t)
{
  os << "Thing(" << t.explainSelf() << ")";
  return os;
}

void func3()
{
  throw Thing("Data Corrupt") << " Low level error" << std::endl;
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
  catch (Exception& e) {
      throw Exception("InfiniteLoop","In func2",e) << "Gave up";
  }
  
}

int main()
{
  try {
    func1();
  }
  catch (Exception& e) {
    std::cerr << "*** main caught Exception, output is ***\n"
	 << "(" << e.explainSelf() << ")"
	 << "*** After exception output ***"
	 << std::endl;

    std::cerr << "\nCategory name list:\n";

#if 0
    if(e.explainSelf() != answer) {
      std::cerr << "not right answer\n(" << answer << ")\n"
	   << std::endl;
      abort();
    }
#endif

  }
  return 0; 
}
