#ifndef FWCore_Utilities_SoATupleHelper_h
#define FWCore_Utilities_SoATupleHelper_h
// -*- C++ -*-
//
// Package:     Package
// Class  :     SoATupleHelper
// 
/**\class SoATupleHelper SoATupleHelper.h "SoATupleHelper.h"

 Description: Helper classes for SoATuple

 Usage:
    These classes are an internal detail of SoATuple.

*/
//
// Original Author:  Chris Jones
//         Created:  Tue, 16 Apr 2013 21:06:08 GMT
// $Id: SoATupleHelper.h,v 1.4 2013/04/23 20:07:48 chrjones Exp $
//

// system include files
#include <tuple>
#include <algorithm>

// user include files

// forward declarations
namespace edm {
  namespace soahelper {
    
    /**Given a leading memory size, iSizeSoFar, and an alignment boundary requirement of iBoundary, returns how much additional memory padding is needed.
     This function assumes that when iSizeSoFar==0 that we are already properly aligned.*/
    constexpr unsigned int padding_needed(size_t iSizeSoFar, unsigned int iBoundary) {
      return (iBoundary - iSizeSoFar % iBoundary) % iBoundary;
    }
    
    /**
     Given a starting value of I=0, and the index to the argument you want to get, J,
     will recursively step through the arguments, Args, until it hits the one you want, Jth, and then will return it as return type Ret.
     */
    
    template<unsigned int I, unsigned int J, typename Ret, typename... Args>
    struct arg_puller;
    
    /**Given a variable number of arguments, returns the 'J'th one. The first caller must use I=0.*/
    template<unsigned int I, unsigned int J, typename Ret, typename F, typename... Args>
    struct arg_puller<I,J,Ret,F, Args...> {
      static Ret pull(F const&, const Args&... args) {
        return arg_puller<I+1, J,Ret,Args...>::pull(args...);
      }
    };
    
    /**End condition of the template recursion */
    template<unsigned int I, typename Ret, typename F, typename... Args>
    struct arg_puller<I,I,Ret,F, Args...> {
      static Ret pull(F const& iV, const Args&...) {
        return iV;
      }
    };

    /** A decorator class used in SoATuple's template argument list to denote that
     the type T should be stored with an unusual byte alignment given by ALIGNMENT.
     NOTE: It is up to the user to be sure sizeof(T) >= ALIGNMENT.
     */
    template<typename T, unsigned int ALIGNMENT>
    struct Aligned {
      static const unsigned int kAlignment = ALIGNMENT;
      typedef T Type;
    };
    
    
    /** Class used by SoATupleHelper to determine the proper alignment of the requested type.
     The default is to just use 'alignof(T)'. Users can change the alignment by instead using
     Aligned<T,ALIGNMENT> for which AlignmentHelper has a specialization.
     */
    template<typename T>
    struct AlignmentHelper {
      static const std::size_t kAlignment = alignof(T);
      typedef T Type;
    };

    /** Specialization of ALignmentHelper for Aligned<T, ALIGNMENT>. This allows users
     to specify non-default alignment values for the internal arrays of SoATuple.
     */
    template<typename T, unsigned int ALIGNMENT>
    struct AlignmentHelper<Aligned<T,ALIGNMENT>> {
      static const std::size_t kAlignment = ALIGNMENT;
      typedef T Type;
    };
    
    /**Implements most of the internal functions used by SoATuple. The argument I is used to recursively step
     through each arugment Args when doing the work. SoATupleHelper<I,Args> actually operates on the I-1 argument.
     There is a specialization of SoATulpeHelper with I=0 which is used to stop the template recursion.
     */
    template<unsigned int I, typename... Args>
    struct SoATupleHelper
    {
      typedef AlignmentHelper<typename std::tuple_element<I-1, std::tuple<Args...>>::type> AlignmentInfo;
      typedef typename AlignmentInfo::Type Type;
      typedef SoATupleHelper<I-1,Args...> NextHelper;

      static const std::size_t max_alignment = AlignmentInfo::kAlignment > NextHelper::max_alignment ?
                                               AlignmentInfo::kAlignment: NextHelper::max_alignment;
      
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
      static const std::size_t max_alignment = 0;
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
      size_t usedSoFar = NextHelper::moveToNew(iNewMemory,iSize, iReserve, oToSet);
      
      //find new start
      const unsigned int boundary = AlignmentInfo::kAlignment;
      
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
      size_t usedSoFar = NextHelper::copyToNew(iNewMemory,iSize, iReserve, iFrom, oToSet);
      
      //find new start
      const unsigned int boundary = AlignmentInfo::kAlignment;
      
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
      size_t usedSoFar = NextHelper::spaceNeededFor(iNElements);
      const unsigned int boundary = AlignmentInfo::kAlignment;
      unsigned int additionalSize = padding_needed(usedSoFar,boundary) + iNElements*sizeof(Type);
      return usedSoFar+additionalSize;
    }
    
    template<unsigned int I, typename... Args>
    void SoATupleHelper<I,Args...>::push_back(void** iToSet, size_t iSize, std::tuple<Args...> const& iValues) {
      new (static_cast<Type*>(*(iToSet+I-1))+iSize) Type(std::get<I-1>(iValues));
      
      NextHelper::push_back(iToSet,iSize,iValues);
    }

    template<unsigned int I, typename... Args>
    template<typename ... FArgs>
    void SoATupleHelper<I,Args...>::emplace_back(void** iToSet, size_t iSize, FArgs... iValues) {
      new (static_cast<Type*>(*(iToSet+I-1))+iSize) Type(arg_puller<0,I-1,Type const&, FArgs...>::pull(std::forward<FArgs>(iValues)...));
      
      NextHelper::emplace_back(iToSet,iSize,std::forward<FArgs>(iValues)...);
    }

    template<unsigned int I, typename... Args>
    void SoATupleHelper<I,Args...>::destroy(void** iToSet, size_t iSize) {
      void** start = iToSet+I-1;
      Type* values = static_cast<Type*>(*start);
      
      for(auto it = values; it != values+iSize; ++it) {
        it->~Type();
      }
      
      NextHelper::destroy(iToSet,iSize);
    }

  }
}

#endif
