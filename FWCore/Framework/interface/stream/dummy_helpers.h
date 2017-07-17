#ifndef Subsystem_Package_dummy_helpers_h
#define Subsystem_Package_dummy_helpers_h
// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     dummy_helpers
// 
/**\class dummy_helpers dummy_helpers.h "dummy_helpers.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sat, 03 Aug 2013 21:42:38 GMT
//

// system include files
#include <memory>
#include <vector>
#include <mutex>
// user include files

// forward declarations

namespace edm {
  namespace stream {
    namespace impl {
      struct dummy_ptr {
        void * get() { return nullptr;}
        void reset(void*) {}
        void * release() { return nullptr;}
      };
      
      struct dummy_vec {
        void resize(size_t) {}
        dummy_ptr operator[](unsigned int) { return dummy_ptr();}
      };
      
      struct dummy_mutex {
        void lock() {}
        void unlock() {}
      };
      
      template<typename T>
      struct choose_unique_ptr {
        typedef std::unique_ptr<T> type;
      };
      template<>
      struct choose_unique_ptr<void> {
        typedef dummy_ptr type;
      };
      
      template<>
      struct choose_unique_ptr<void const> {
        typedef dummy_ptr type;
      };
      
      template<typename T>
      struct choose_shared_vec {
        typedef std::vector<std::shared_ptr<T>> type;
      };
      template<>
      struct choose_shared_vec<void> {
        typedef dummy_vec type;
      };
      template<>
      struct choose_shared_vec<void const> {
        typedef dummy_vec type;
      };
      template<typename T>
      struct choose_mutex {
        using type = std::mutex;
      };
      template<>
      struct choose_mutex<void> {
        using type = dummy_mutex;
      };
    }
  }
}


#endif
