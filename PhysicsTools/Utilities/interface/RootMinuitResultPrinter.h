#ifndef PhysicsTools_Utilities_RootMinuitResultPrinter_h
#define PhysicsTools_Utilities_RootMinuitResultPrinter_h
#include <iostream>

namespace fit {

  template<typename Function>
  struct RootMinuitResultPrinter {
    static void print(double amin, unsigned int numberOfFreeParameters, const Function & f) {
      std::cout << "minimum function = " << amin << ", free parameters = " << numberOfFreeParameters
		<< std::endl;      
    }
  };

}

#endif
