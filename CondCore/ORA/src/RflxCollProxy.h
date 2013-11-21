#ifndef CONDCORE_ORA_RFLXCOLLPROXY
#define CONDCORE_ORA_RFLXCOLLPROXY 1

#include <iostream>

namespace Reflex {

  class CollFuncTable {
    public:
        int first_func() { std::cerr << "ERROR not yet implemented !!!" <<std::endl; return 0; }
        int next_func () { std::cerr << "ERROR not yet implemented !!!" <<std::endl; return 0; }
        int size_func () { std::cerr << "ERROR not yet implemented !!!" <<std::endl; return 0; }
        int feed_func () { std::cerr << "ERROR not yet implemented !!!" <<std::endl; return 0; }
        int clear_func() { std::cerr << "ERROR not yet implemented !!!" <<std::endl; return 0; }

  }; // end class CollFuncTable

} // end namespace Reflex

#endif // CONDCORE_ORA_RFLXCOLLPROXY
