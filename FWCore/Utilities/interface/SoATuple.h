#ifndef FWCore_Framework_SoATuple_h
#define FWCore_Framework_SoATuple_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     SoATuple
// 
/**\class SoATuple SoATuple.h "FWCore/Framework/interface/SoATuple.h"

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
*/
//
// Original Author:  Chris Jones
//         Created:  Tue, 16 Apr 2013 20:34:31 GMT
// $Id$
//

// system include files
#include <algorithm>
#include <tuple>

// user include files
#include "FWCore/Utilities/interface/SoATupleHelper.h"

// forward declarations

namespace edm {
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
      delete [] static_cast<char*>(m_values[0]);
    }
    
    // ---------- const member functions ---------------------
    size_t size() const { return m_size;}
    size_t capacity() const {return m_reserved;}

    /** Returns const access to data element I of item iIndex */
    template<unsigned int I>
    typename std::tuple_element<I, std::tuple<Args...>>::type const& get(unsigned int iIndex) const {
      typedef typename std::tuple_element<I, std::tuple<Args...>>::type ReturnType;
      return *(static_cast<ReturnType const*>(m_values[I])+iIndex);
    }

    /** Returns the beginning of the container holding all Ith data elements*/
    template<unsigned int I>
    typename std::tuple_element<I, std::tuple<Args...>>::type const* begin() const {
      typedef typename std::tuple_element<I, std::tuple<Args...>>::type ReturnType;
      return static_cast<ReturnType const*>(m_values[I]);
    }
    /** Returns the end of the container holding all Ith data elements*/
    template<unsigned int I>
    typename std::tuple_element<I, std::tuple<Args...>>::type const* end() const {
      typedef typename std::tuple_element<I, std::tuple<Args...>>::type ReturnType;
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
    void emplace_back(FArgs... values) {
      if(size()+1>capacity()) {
        reserve(size()*2+1);
      }
      soahelper::SoATupleHelper<sizeof...(Args),Args...>::emplace_back(m_values,m_size,std::forward<Args>(values)...);
      ++m_size;
    }

    /** Returns access to data element I of item iIndex */
    template<unsigned int I>
    typename std::tuple_element<I, std::tuple<Args...>>::type& get(unsigned int iIndex) {
      typedef typename std::tuple_element<I, std::tuple<Args...>>::type ReturnType;
      return *(static_cast<ReturnType*>(m_values[I])+iIndex);
    }
    
    /** Returns the beginning of the container holding all Ith data elements*/
    template<unsigned int I>
    typename std::tuple_element<I, std::tuple<Args...>>::type* begin() {
      typedef typename std::tuple_element<I, std::tuple<Args...>>::type ReturnType;
      return static_cast<ReturnType*>(m_values[I]);
    }
    /** Returns the end of the container holding all Ith data elements*/
    template<unsigned int I>
    typename std::tuple_element<I, std::tuple<Args...>>::type* end() {
      typedef typename std::tuple_element<I, std::tuple<Args...>>::type ReturnType;
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
      size_t memoryNeeded = soahelper::SoATupleHelper<sizeof...(Args),Args...>::spaceNeededFor(iToSize);
      //are there alignment issues?
      char * newMemory = new char[memoryNeeded];
      void * oldMemory =m_values[0];
      soahelper::SoATupleHelper<sizeof...(Args),Args...>::moveToNew(newMemory,m_size, iToSize, m_values);
      m_reserved = iToSize;
      delete [] static_cast<char*>(oldMemory);
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
