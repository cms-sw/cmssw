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

    template <unsigned int I, unsigned int J, typename Ret, typename... Args>
    struct arg_puller;

    /**Given a variable number of arguments, returns the 'J'th one. The first caller must use I=0.*/
    template <unsigned int I, unsigned int J, typename Ret, typename F, typename... Args>
    struct arg_puller<I, J, Ret, F, Args...> {
      static constexpr Ret pull(F const&, const Args&... args) {
        return arg_puller<I + 1, J, Ret, Args...>::pull(args...);
      }
    };

    /**End condition of the template recursion */
    template <unsigned int I, typename Ret, typename F, typename... Args>
    struct arg_puller<I, I, Ret, F, Args...> {
      static constexpr Ret pull(F const& iV, const Args&...) { return iV; }
    };

    /** A decorator class used in SoATuple's template argument list to denote that
     the type T should be stored with an unusual byte alignment given by ALIGNMENT.
     NOTE: It is up to the user to be sure sizeof(T) >= ALIGNMENT.
     */
    template <typename T, unsigned int ALIGNMENT>
    struct Aligned {
      static constexpr unsigned int kAlignment = ALIGNMENT;
      using Type = T;
    };

    /** Class used by SoATupleHelper to determine the proper alignment of the requested type.
     The default is to just use 'alignof(T)'. Users can change the alignment by instead using
     Aligned<T,ALIGNMENT> for which AlignmentHelper has a specialization.
     */
    template <typename T>
    struct AlignmentHelper {
      static constexpr std::size_t kAlignment = alignof(T);
      using Type = T;
    };

    /** Specialization of ALignmentHelper for Aligned<T, ALIGNMENT>. This allows users
     to specify non-default alignment values for the internal arrays of SoATuple.
     */
    template <typename T, unsigned int ALIGNMENT>
    struct AlignmentHelper<Aligned<T, ALIGNMENT>> {
      static constexpr std::size_t kAlignment = ALIGNMENT;
      using Type = T;
    };

    template <size_t I, typename... Args>
    constexpr auto nullPtrType() -> typename std::tuple_element<I, std::tuple<Args...>>::type* {
      return nullptr;
    }

    /**Implements most of the internal functions used by SoATuple.*/
    template <typename... Args>
    struct SoATupleHelper {
      static constexpr std::size_t max_alignment = std::max({AlignmentHelper<Args>::kAlignment...});
      static constexpr unsigned int kNTypes = sizeof...(Args);

      // ---------- static member functions --------------------
      static size_t moveToNew(std::byte* iNewMemory, size_t iSize, size_t iReserve, void** oToSet);
      static size_t copyToNew(std::byte* iNewMemory, size_t iSize, size_t iReserve, void* const* iFrom, void** oToSet);
      static size_t spaceNeededFor(unsigned int iNElements);
      static void push_back(void** iToSet, size_t iSize, std::tuple<Args...> const& iValues);
      template <typename... FArgs>
      static void emplace_back(void** iToSet, size_t iSize, FArgs... iValues);
      static void destroy(void** iToSet, size_t iSize);

      // ---------- member functions ---------------------------
      SoATupleHelper(const SoATupleHelper&) = delete;  // stop default
      SoATupleHelper(SoATupleHelper&&) = delete;       // stop default

      SoATupleHelper& operator=(const SoATupleHelper&) = delete;  // stop default
      SoATupleHelper& operator=(SoATupleHelper&&) = delete;       // stop default
    };

    template <typename... Args>
    size_t SoATupleHelper<Args...>::moveToNew(std::byte* iNewMemory, size_t iSize, size_t iReserve, void** oToSet) {
      size_t usedSoFar = 0;
      auto do_move = [&]<typename T>(T const*, auto I) {
        using Type = typename AlignmentHelper<T>::Type;
        constexpr unsigned int boundary = AlignmentHelper<T>::kAlignment;
        Type* newStart = reinterpret_cast<Type*>(iNewMemory + usedSoFar + padding_needed(usedSoFar, boundary));
        void** oldStart = oToSet + I;
        Type* oldValues = static_cast<Type*>(*oldStart);
        if (oldValues != nullptr) {
          auto ptr = newStart;
          for (auto it = oldValues; it != oldValues + iSize; ++it, ++ptr) {
            new (ptr) Type(std::move(*it));
          }
          for (auto it = oldValues; it != oldValues + iSize; ++it) {
            it->~Type();
          }
        }
        *oldStart = newStart;
        unsigned int additionalSize = padding_needed(usedSoFar, boundary) + iReserve * sizeof(Type);
        usedSoFar += additionalSize;
      };
      //The use of index_sequence gives an index for each type in Args...
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        //For each type, call do_move
        (do_move(nullPtrType<Is, Args...>(), std::integral_constant<size_t, Is>()), ...);
      }(std::make_index_sequence<kNTypes>());
      return usedSoFar;
    }

    template <typename... Args>
    size_t SoATupleHelper<Args...>::copyToNew(
        std::byte* iNewMemory, size_t iSize, size_t iReserve, void* const* iFrom, void** oToSet) {
      size_t usedSoFar = 0;
      auto do_copy = [&]<typename T>(T const*, auto I) {
        using Type = typename AlignmentHelper<T>::Type;
        constexpr unsigned int boundary = AlignmentHelper<T>::kAlignment;
        Type* newStart = reinterpret_cast<Type*>(iNewMemory + usedSoFar + padding_needed(usedSoFar, boundary));
        void* const* oldStart = iFrom + I;
        Type* oldValues = static_cast<Type*>(*oldStart);
        if (oldValues != nullptr) {
          auto ptr = newStart;
          for (auto it = oldValues; it != oldValues + iSize; ++it, ++ptr) {
            new (ptr) Type(*it);
          }
        }
        *(oToSet + I) = newStart;
        unsigned int additionalSize = padding_needed(usedSoFar, boundary) + iReserve * sizeof(Type);
        usedSoFar += additionalSize;
      };
      //The use of index_sequence gives an index for each type in Args...
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        //For each type, call do_copy
        (do_copy(nullPtrType<Is, Args...>(), std::integral_constant<size_t, Is>()), ...);
      }(std::make_index_sequence<kNTypes>());
      return usedSoFar;
    }

    template <typename... Args>
    size_t SoATupleHelper<Args...>::spaceNeededFor(unsigned int iNElements) {
      size_t usedSoFar = 0;
      auto do_space = [&]<typename T>(T const*) {
        constexpr unsigned int boundary = AlignmentHelper<T>::kAlignment;
        using Type = typename AlignmentHelper<T>::Type;
        unsigned int additionalSize = padding_needed(usedSoFar, boundary) + iNElements * sizeof(Type);
        usedSoFar += additionalSize;
      };
      //The use of index_sequence gives an index for each type in Args...
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        //For each type, call do_space
        (do_space(nullPtrType<Is, Args...>()), ...);
      }(std::make_index_sequence<kNTypes>());
      return usedSoFar;
    }

    template <typename... Args>
    void SoATupleHelper<Args...>::push_back(void** iToSet, size_t iSize, std::tuple<Args...> const& iValues) {
      auto do_placement = [&]<typename T>(auto Index, T const*) {
        using Type = typename AlignmentHelper<T>::Type;
        new (static_cast<Type*>(*(iToSet + Index)) + iSize) Type(std::get<Index>(iValues));
      };
      //The use of index_sequence gives an index for each type in Args...
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        //For each type, call do_placement
        (do_placement(std::integral_constant<size_t, Is>(), nullPtrType<Is, Args...>()), ...);
      }(std::make_index_sequence<kNTypes>());
    }

    template <typename... Args>
    template <typename... FArgs>
    void SoATupleHelper<Args...>::emplace_back(void** iToSet, size_t iSize, FArgs... iValues) {
      auto do_placement = [&]<typename T>(auto Index, T const*) {
        using Type = typename AlignmentHelper<T>::Type;
        new (static_cast<Type*>(*(iToSet + Index)) + iSize)
            Type(arg_puller<0, Index, Type const&, FArgs...>::pull(std::forward<FArgs>(iValues)...));
      };
      //The use of index_sequence gives an index for each type in Args...
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        //For each type, call do_placement
        (do_placement(std::integral_constant<size_t, Is>(), nullPtrType<Is, Args...>()), ...);
      }(std::make_index_sequence<kNTypes>());
    }

    template <typename... Args>
    void SoATupleHelper<Args...>::destroy(void** iToSet, size_t iSize) {
      auto do_destroy = [&]<typename T>(std::size_t I, T const*) {
        void** start = iToSet + I;
        using Type = typename AlignmentHelper<T>::Type;
        Type* values = static_cast<Type*>(*start);

        for (auto it = values; it != values + iSize; ++it) {
          it->~Type();
        }
      };
      //The use of index_sequence gives an index for each type in Args...
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        //For each type, call do_destroy
        ((do_destroy(Is, nullPtrType<Is, Args...>())), ...);
      }(std::make_index_sequence<kNTypes>());
    }

  }  // namespace soahelper
}  // namespace edm

#endif
