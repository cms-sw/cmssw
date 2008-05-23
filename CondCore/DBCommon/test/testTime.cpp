#include "CondCore/DBCommon/interface/Time.h"


#include<iostream>
int main() {
  using namespace cond;
  for (size_t i=0; i<TIMETYPE_LIST_MAX; i++) 
    std::cout << "Time Specs:" 
	      << " enum " << timeTypeSpecs[i].type
	      << ", name " << timeTypeSpecs[i].name
	      << ", begin " << timeTypeSpecs[i].beginValue
	      << ", end " << timeTypeSpecs[i].endValue
	      << ", invalid " << timeTypeSpecs[i].invalidValue
	      << std::endl;

  try {
    for (size_t i=0; i<TIMETYPE_LIST_MAX; i++)
      if (cond::findSpecs(timeTypeSpecs[i].name).type!=timeTypeSpecs[i].type)
	std::cout << "error in find for " << timeTypeSpecs[i].name << std::endl;
    
    cond::findSpecs("fake");
  }
  catch(cms::Exception const & e) {
    std::cout << e.what() << std::endl;
  }

  return 0;
}
