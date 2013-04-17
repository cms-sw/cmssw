#ifndef Subsystem_Package_SoATupleHelper_h
#define Subsystem_Package_SoATupleHelper_h
// -*- C++ -*-
//
// Package:     Package
// Class  :     SoATupleHelper
// 
/**\class SoATupleHelper SoATupleHelper.h "SoATupleHelper.h"

 Description: Helper class for SoATuple

 Usage:
    This class is an internal detail of SoATuple.

*/
//
// Original Author:  Chris Jones
//         Created:  Tue, 16 Apr 2013 21:06:08 GMT
// $Id$
//

// system include files
#include <tuple>
#include <algorithm>

// user include files

// forward declarations
namespace edm {
  namespace soahelper {
    
    constexpr unsigned int alignment_boundary_impl(size_t iSize, size_t iCheckSize)  {
      return 0 == iSize % iCheckSize ? iCheckSize : alignment_boundary_impl(iSize,iCheckSize/2);
    }
    
    /**Given a type of size iSize, returns the smallest memory alignment boundary required for the type */
    constexpr unsigned int alignment_boundary(size_t iSize) {
      return alignment_boundary_impl(iSize, sizeof(char*));
    }
    
    /**Given a leading memory size, iSizeSoFar, and an alignment boundary requirement of iBoundary, returns how much additional memory padding is needed.
     This function assumes that when iSizeSoFar==0 that we are already properly aligned.*/
    constexpr unsigned int padding_needed(size_t iSizeSoFar, unsigned int iBoundary) {
      return (iBoundary - iSizeSoFar % iBoundary) % iBoundary;
    }
    
    /**Given a variable number of arguments, returns the 'J'th one. The first caller must use I=0.*/
    template<unsigned int I, unsigned int J, typename Ret, typename F, typename... Args>
    struct arg_puller{
      static Ret pull(F, Args... args) {
        return arg_puller<I+1, J,Ret,Args...>::pull(std::forward<Args>(args)...);
      }
    };
    
    template<unsigned I, typename Ret, typename F, typename... Args>
    struct arg_puller<I,I,Ret,F, Args...> {
      static Ret pull(F const& iV, Args...) {
        return iV;
      }
    };



    template<unsigned int I, typename... Args>
    struct SoATupleHelper
    {
      
      // ---------- static member functions --------------------
      static size_t moveToNew(char* iNewMemory, size_t iSize, size_t iReserve, void** oToSet);
      static size_t copyToNew(char* iNewMemory, size_t iSize, size_t iReserve, void* const* iFrom, void** oToSet);
      static size_t spaceNeededFor(unsigned int iNElements);
      static void push_back(void** iToSet, size_t iSize, std::tuple<Args...> const& iValues);
      template<typename... FArgs>
      static void emplace_back(void** iToSet, size_t iSize, FArgs... iValues);
      static void destroy(void** iToSet, size_t iSize);
      
      // ---------- member functions ---------------------------
      SoATupleHelper(const SoATupleHelper&) = delete; // stop default
      
      const SoATupleHelper& operator=(const SoATupleHelper&) = delete; // stop default
      
    };
    
    //Specialization used to stop recursion
    template<typename... Args>
    struct SoATupleHelper<0,Args...> {
      static void destroy(void** /*iToSet*/, size_t /*iSize*/) {
      }
      
      static void push_back(void** /*iToSet*/, size_t /*iSize*/, std::tuple<Args...> const& /*values*/) {
      }

      template<typename... FArgs>
      static void emplace_back(void** iToSet, size_t iSize, FArgs... iValues) {}

      static size_t spaceNeededFor(unsigned int /*iNElements*/) {
        return 0;
      }
      
      static size_t moveToNew(char* /*iNewMemory*/, size_t /*iSize*/, size_t /*iReserve*/, void** /*oToSet*/) {
        return 0;
      }
      
      static size_t copyToNew(char* /*iNewMemory*/, size_t /*iSize*/, size_t /*iReserve*/, void* const* /*iFrom*/, void** /*oToSet*/) {
        return 0;
      }
    };

    template<unsigned int I, typename... Args>
    size_t SoATupleHelper<I,Args...>::moveToNew(char* iNewMemory, size_t iSize, size_t iReserve, void** oToSet) {
      size_t usedSoFar = SoATupleHelper<I-1,Args...>::moveToNew(iNewMemory,iSize, iReserve, oToSet);
      
      typedef typename std::tuple_element<I-1, std::tuple<Args...>>::type Type;
      //find new start
      const unsigned int boundary = alignment_boundary(sizeof(Type));
      
      Type* newStart = reinterpret_cast<Type*>(iNewMemory+usedSoFar+padding_needed(usedSoFar,boundary));
      
      void** oldStart = oToSet+I-1;
      
      Type* oldValues = static_cast<Type*>(*oldStart);
      if(oldValues != nullptr ) {
        auto ptr = newStart;
        for(auto it = oldValues; it != oldValues+iSize; ++it,++ptr) {
          new (ptr) Type(std::move(*it));
        }
        for(auto it = oldValues; it != oldValues+iSize; ++it) {
          it->~Type();
        }
      }
      *oldStart = newStart;
      unsigned int additionalSize = padding_needed(usedSoFar,boundary) + iReserve*sizeof(Type);
      return usedSoFar+additionalSize;
    }

    template<unsigned int I, typename... Args>
    size_t SoATupleHelper<I,Args...>::copyToNew(char* iNewMemory, size_t iSize, size_t iReserve, void* const* iFrom, void** oToSet) {
      size_t usedSoFar = SoATupleHelper<I-1,Args...>::copyToNew(iNewMemory,iSize, iReserve, iFrom, oToSet);
      
      typedef typename std::tuple_element<I-1, std::tuple<Args...>>::type Type;
      //find new start
      const unsigned int boundary = alignment_boundary(sizeof(Type));
      
      Type* newStart = reinterpret_cast<Type*>(iNewMemory+usedSoFar+padding_needed(usedSoFar,boundary));
      
      void* const* oldStart = iFrom+I-1;
      
      Type* oldValues = static_cast<Type*>(*oldStart);
      if(oldValues != nullptr ) {
        auto ptr = newStart;
        for(auto it = oldValues; it != oldValues+iSize; ++it,++ptr) {
          new (ptr) Type(*it);
        }
      }
      *(oToSet+I-1) = newStart;
      unsigned int additionalSize = padding_needed(usedSoFar,boundary) + iReserve*sizeof(Type);
      return usedSoFar+additionalSize;
    }

    
    template<unsigned int I, typename... Args>
    size_t SoATupleHelper<I,Args...>::spaceNeededFor(unsigned int iNElements) {
      size_t usedSoFar = SoATupleHelper<I-1,Args...>::spaceNeededFor(iNElements);
      typedef typename std::tuple_element<I-1, std::tuple<Args...>>::type Type;
      const unsigned int boundary = alignment_boundary(sizeof(Type));
      unsigned int additionalSize = padding_needed(usedSoFar,boundary) + iNElements*sizeof(Type);
      return usedSoFar+additionalSize;
    }
    
    template<unsigned int I, typename... Args>
    void SoATupleHelper<I,Args...>::push_back(void** iToSet, size_t iSize, std::tuple<Args...> const& iValues) {
      typedef typename std::tuple_element<I-1, std::tuple<Args...>>::type Type;
      new (static_cast<Type*>(*(iToSet+I-1))+iSize) Type(std::get<I-1>(iValues));
      
      SoATupleHelper<I-1,Args...>::push_back(iToSet,iSize,iValues);
    }

    template<unsigned int I, typename... Args>
    template<typename ... FArgs>
    void SoATupleHelper<I,Args...>::emplace_back(void** iToSet, size_t iSize, FArgs... iValues) {
      typedef typename std::tuple_element<I-1, std::tuple<Args...>>::type Type;
      new (static_cast<Type*>(*(iToSet+I-1))+iSize) Type(arg_puller<0,I-1,Type const&, Args...>::pull(iValues...));
      
      SoATupleHelper<I-1,Args...>::emplace_back(iToSet,iSize,iValues...);
    }

    template<unsigned int I, typename... Args>
    void SoATupleHelper<I,Args...>::destroy(void** iToSet, size_t iSize) {
      typedef typename std::tuple_element<I-1, std::tuple<Args...>>::type Type;
      void** start = iToSet+I-1;
      Type* values = static_cast<Type*>(*start);
      
      for(auto it = values; it != values+iSize; ++it) {
        it->~Type();
      }
      
      SoATupleHelper<I-1,Args...>::destroy(iToSet,iSize);
    }

  }
}

#endif
