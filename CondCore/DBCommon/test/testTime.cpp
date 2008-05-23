#include "CondCore/DBCommon/interface/Time.h"


#include<iostream>
int main() {
  using namespace cond;
  for (size_t i=0; i<TIMETYPE_LIST_MAX; i++) 
    std::cout << "Time Specs:" 
	      << " enum " << timeTypeSpec[i].type
	      << ", name " << timeTypeSpec[i].name
	      << ", begin " << timeTypeSpec[i].beginValue
	      << ", end " << timeTypeSpec[i].endValue
	      << ", invalid " << timeTypeSpec[i].invalidValue
	      << std::endl;

  try {
  for (size_t i=0; i<TIMETYPE_LIST_MAX; i++) 
    if (cond::findSpecs(timeTypeSpec[i].name).type!=timeTypeSpec[i].type)
      std::cout << "error in find for " << timeTypeSpec[i].name std::endl;
  
  cond::findSpecs("fake");
  }
  catch(cms::Exception const & e) {
    std::cout << e.what() << std::endl;
  }

  return 0;
}
