#ifndef FWCore_Utilities_SoATuple_h
#define FWCore_Utilities_SoATuple_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     SoATuple
// 
/**\class SoATuple SoATuple.h "FWCore/Utilities/interface/SoATuple.h"

 Description: Structure of Arrays Tuple

 Usage:
    Often we want to group data which are related to one item and then group related items. 
 This is often done by making a structure,Foo, to hold the data related to one item and then place
 the structure into a container (e.g. std::vector<Foo>). This is referred to as an 'Array of Structures'.
 However, this grouping can be inefficient if not all of the data about one item are used at the same
 time. So to access one data for one item will cause the CPU to retrieve nearby memory which is then
 not used. If the code is looking at the same data for multiple items this will lead to many data cache
 misses in the CPU.
 
 A different grouping is to place the first data element for all items into one array, the second data
 element for all items into a second array, then so on. This is referred to as a 'Structure of Arrays'.

 This class will take an arbitrary number of template arguments and it will group data of that argument
 type together in memory.
 
Example: Data about one item is represented by a double, int and bool
\code
    edm::SoATuple<double,int, bool> s;
\endcode

One can then push data into the collection. You must insert all data for each item.
\code
    s.push_back(std::make_tuple(double{3.14},int{5},false));
    s.emplace_back(double{90.},int{0},true);
\endcode
It is best if you call 'reserve' before pushing or emplacing items into the container in order to
minimize memory allocations.
 
 
You get the data out by specify the 'ordinal' of the data element as well as the index
\code
    double v = s.get<0>(1); //this return 90.
\encode

It is possible to loop over a data element for all items in the collection
\code
    for(auto it = s.begin<1>(), itEnd=s.end<1>(); it != itEnd; ++it) {
      std::cout<<*it<<" ";
    }
    //This returns '5 0 '
\encode
 
This template arguments for this class are not limited to simple builtins, any type can be used:
\code
    edm::SoATuple<std::string, ThreeVector> sComplex;
\endcode
 
To help keep track of the purpose of the template arguments, we suggest using an enum to denote each one:
\code
    enum {kPx,kPy,kPz};
    edm::SoATuple<double,double,double> s3Vecs;
    ...
    if(s.3Vecs.get<kPx>(i) > s.3Vecs.get<kPy>(i)) { ... }
\endcode

A non-default alignment for a stored type can be specified by using the edm::Aligned<T,I> where I is an
 unsigned int value denoting the requested byte alignment. There is also a specialized version, edm::AlignedVec
 which has the proper alignment for SSE operations (16 byte aligned).
\code
    edm::SoATuple<edm::Aligned<float,16>,edm::Aligned<float,16>,edm::Aligned<float,16>> vFloats;
\endcode
which is equivalent to
\code
    edm::SoATuple<edm::AlignedVec<float>,edm::AlignedVec<float>,edm::AlignedVec<float>> vFloats;
\endcode
 
Explicitly aligned types and defaultly aligned types can be freely mixed in any order within the template arguments.
 
*/
//
// Original Author:  Chris Jones
//         Created:  Tue, 16 Apr 2013 20:34:31 GMT
// $Id: SoATuple.h,v 1.4 2013/04/23 20:07:48 chrjones Exp $
//

// system include files
#include <algorithm>
#include <tuple>
#include <cassert>

// user include files
#include "FWCore/Utilities/interface/SoATupleHelper.h"
#include "FWCore/Utilities/interface/GCCPrerequisite.h"


// forward declarations

namespace edm {

  //The class Aligned is used to specify a non-default alignment for a class
  using edm::soahelper::Aligned;
  
  //Proper alignment for doing vectorized operations on CPU
  template<typename T> using AlignedVec = Aligned<T,16>;
  
  template <typename... Args>
  class SoATuple
  {
    
  public:
    typedef typename std::tuple<Args...> element;

    SoATuple(): m_size(0),m_reserved(0){
      for(auto& v : m_values) {
        v = nullptr;
      }
    }
    SoATuple(const SoATuple<Args...>& iOther):m_size(0),m_reserved(0) {
      for(auto& v : m_values) {
        v = nullptr;
      }
      reserve(iOther.m_size);
      soahelper::SoATupleHelper<sizeof...(Args),Args...>::copyToNew(static_cast<char*>(m_values[0]),iOther.m_size,m_reserved,iOther.m_values,m_values);
      m_size = iOther.m_size;
    }

    SoATuple(SoATuple<Args...>&& iOther):m_size(0),m_reserved(0) {
      for(auto& v : m_values) {
        v = nullptr;
      }
      this->swap(iOther);
    }

    const SoATuple<Args...>& operator=(const SoATuple<Args...>& iRHS) {
      SoATuple<Args...> temp(iRHS);
      this->swap(temp);
      return *this;
    }

    ~SoATuple() {
      soahelper::SoATupleHelper<sizeof...(Args),Args...>::destroy(m_values,m_size);
      typedef std::aligned_storage<soahelper::SoATupleHelper<sizeof...(Args),Args...>::max_alignment,
      soahelper::SoATupleHelper<sizeof...(Args),Args...>::max_alignment> AlignedType;

      delete [] static_cast<AlignedType*>(m_values[0]);
    }
    
    // ---------- const member functions ---------------------
    size_t size() const { return m_size;}
    size_t capacity() const {return m_reserved;}

    /** Returns const access to data element I of item iIndex */
    template<unsigned int I>
    typename soahelper::AlignmentHelper<typename std::tuple_element<I, std::tuple<Args...>>::type>::Type const& get(unsigned int iIndex) const {
      typedef typename soahelper::AlignmentHelper<typename std::tuple_element<I, std::tuple<Args...>>::type>::Type ReturnType;
      return *(static_cast<ReturnType const*>(m_values[I])+iIndex);
    }

    /** Returns the beginning of the container holding all Ith data elements*/
    template<unsigned int I>
    typename soahelper::AlignmentHelper<typename std::tuple_element<I, std::tuple<Args...>>::type>::Type const* begin() const {
      typedef soahelper::AlignmentHelper<typename std::tuple_element<I, std::tuple<Args...>>::type> Helper;
      typedef typename Helper::Type ReturnType;
#if GCC_PREREQUISITE(4,7,0)
      return static_cast<ReturnType const*>(__builtin_assume_aligned(m_values[I],Helper::kAlignment));
#else
      return static_cast<ReturnType const*>(m_values[I]);
#endif
    }
    /** Returns the end of the container holding all Ith data elements*/
    template<unsigned int I>
    typename soahelper::AlignmentHelper<typename std::tuple_element<I, std::tuple<Args...>>::type>::Type const* end() const {
      typedef typename soahelper::AlignmentHelper<typename std::tuple_element<I, std::tuple<Args...>>::type>::Type ReturnType;
      return static_cast<ReturnType const*>(m_values[I])+m_size;
    }

    // ---------- member functions ---------------------------
    /** Makes sure to hold enough memory to contain at least iToSize entries. */
    void reserve(unsigned int iToSize) {
      if(iToSize > m_reserved) {
        changeSize(iToSize);
      }
    }
    
    /** Shrinks the amount of memory used so as to only have just enough to hold all entries.*/
    void shrink_to_fit() {
      if(m_reserved > m_size) {
        changeSize(m_size);
      }
    }

    /** Adds one entry to the end of the list. Memory grows as needed.*/
    void push_back(element const& values) {
      if(size()+1>capacity()) {
        reserve(size()*2+1);
      }
      soahelper::SoATupleHelper<sizeof...(Args),Args...>::push_back(m_values,m_size,values);
      ++m_size;
    }

    /** Adds one entry to the end of the list. The arguments are used to instantiate each data element in the order defined in the template arguments.*/
    template< typename... FArgs>
    void emplace_back(FArgs&&... values) {
      if(size()+1>capacity()) {
        reserve(size()*2+1);
      }
      soahelper::SoATupleHelper<sizeof...(Args),Args...>::emplace_back(m_values,m_size,std::forward<FArgs>(values)...);
      ++m_size;
    }

    /** Returns access to data element I of item iIndex */
    template<unsigned int I>
    typename soahelper::AlignmentHelper<typename std::tuple_element<I, std::tuple<Args...>>::type>::Type& get(unsigned int iIndex) {
      typedef typename soahelper::AlignmentHelper<typename std::tuple_element<I, std::tuple<Args...>>::type>::Type ReturnType;
      return *(static_cast<ReturnType*>(m_values[I])+iIndex);
    }
    
    /** Returns the beginning of the container holding all Ith data elements*/
    template<unsigned int I>
    typename soahelper::AlignmentHelper<typename std::tuple_element<I, std::tuple<Args...>>::type>::Type* begin() {
      typedef soahelper::AlignmentHelper<typename std::tuple_element<I, std::tuple<Args...>>::type> Helper;
      typedef typename Helper::Type ReturnType;
#if GCC_PREREQUISITE(4,7,0)
      return static_cast<ReturnType*>(__builtin_assume_aligned(m_values[I],Helper::kAlignment));
#else
      return static_cast<ReturnType*>(m_values[I]);
#endif
    }
    /** Returns the end of the container holding all Ith data elements*/
    template<unsigned int I>
    typename soahelper::AlignmentHelper<typename std::tuple_element<I, std::tuple<Args...>>::type>::Type* end() {
      typedef typename soahelper::AlignmentHelper<typename std::tuple_element<I, std::tuple<Args...>>::type>::Type ReturnType;
      return static_cast<ReturnType*>(m_values[I])+m_size;
    }
    
    void swap(SoATuple<Args...>& iOther) {
      std::swap(m_size,iOther.m_size);
      std::swap(m_reserved,iOther.m_reserved);
      for(unsigned int i=0; i<sizeof...(Args);++i) {
        std::swap(m_values[i],iOther.m_values[i]);
      }
    }
  private:
        
    void changeSize(unsigned int iToSize) {
      assert(m_size<=iToSize);
      const size_t memoryNeededInBytes = soahelper::SoATupleHelper<sizeof...(Args),Args...>::spaceNeededFor(iToSize);
      //align memory of the array to be on the strictest alignment boundary for any type in the Tuple
      // This is done by creating an array of a type that has that same alignment restriction and minimum size.
      // This has the draw back of possibly padding the array by one extra element if the memoryNeededInBytes is not
      // a strict multiple of max_alignment.
      // NOTE: using new char[...] would likely cause more padding based on C++11 5.3.4 paragraph 10 where it
      // says the alignment will be for the strictest requirement for an object whose size < size of array. So
      // if the array were for 64 bytes and the strictest requirement of any object was 8 bytes then the entire
      // char array would be aligned on an 8 byte boundary. However, if the SoATuple<char,char> only 1 byte alignment
      // is needed. The following algorithm would require only 1 byte alignment
      const std::size_t max_alignment = soahelper::SoATupleHelper<sizeof...(Args),Args...>::max_alignment;
      typedef std::aligned_storage<soahelper::SoATupleHelper<sizeof...(Args),Args...>::max_alignment,
                                   soahelper::SoATupleHelper<sizeof...(Args),Args...>::max_alignment> AlignedType;
      //If needed, pad the number of items by 1
      const size_t itemsNeeded = (memoryNeededInBytes+max_alignment-1)/sizeof(AlignedType);
      char * newMemory = static_cast<char*>(static_cast<void*>(new AlignedType[itemsNeeded]));
      void * oldMemory =m_values[0];
      soahelper::SoATupleHelper<sizeof...(Args),Args...>::moveToNew(newMemory,m_size, iToSize, m_values);
      m_reserved = iToSize;
      delete [] static_cast<AlignedType*>(oldMemory);
    }
    // ---------- member data --------------------------------
    //Pointers to where each column starts in the shared memory array
    //m_values[0] also points to the beginning of the shared memory area
    void* m_values[sizeof...(Args)];
    size_t m_size;
    size_t m_reserved;
  };
}


#endif
