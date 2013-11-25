#ifndef CONDCORE_ORA_RFLXCOLLPROXY
#define CONDCORE_ORA_RFLXCOLLPROXY 1

#include <iostream>

#include "RflxEnv.h"

namespace Reflex {

  class CollFuncTable {
    public:
        void* first_func(Reflex::Environ<long int>*) { std::cerr << "CondCore/ORA/src/> Reflex::CollFuncTable> ERROR function void* first_func(Reflex::Environ<long int>*) not yet implemented !!!" <<std::endl; return 0; }
        void* next_func (Reflex::Environ<long int>*) { std::cerr << "CondCore/ORA/src/> Reflex::CollFuncTable> ERROR function void* next_func (Reflex::Environ<long int>*) not yet implemented !!!" <<std::endl; return 0; }
        void* size_func (Reflex::Environ<long int>*) { std::cerr << "CondCore/ORA/src/> Reflex::CollFuncTable> ERROR function void* size_func (Reflex::Environ<long int>*) not yet implemented !!!" <<std::endl; return 0; }
        void* feed_func (Reflex::Environ<long int>*) { std::cerr << "CondCore/ORA/src/> Reflex::CollFuncTable> ERROR function void* feed_func (Reflex::Environ<long int>*) not yet implemented !!!" <<std::endl; return 0; }
        void* feed_func (void*&, void*&, int)        { std::cerr << "CondCore/ORA/src/> Reflex::CollFuncTable> ERROR function void* feed_func (void*&, void*&, int)        not yet implemented !!!" <<std::endl; return 0; }
        void* clear_func(Reflex::Environ<long int>*) { std::cerr << "CondCore/ORA/src/> Reflex::CollFuncTable> ERROR function void* clear_func(Reflex::Environ<long int>*) not yet implemented !!!" <<std::endl; return 0; }

  }; // end class CollFuncTable

} // end namespace Reflex

#endif // CONDCORE_ORA_RFLXCOLLPROXY
