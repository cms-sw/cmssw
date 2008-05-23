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

  return 0;
}
