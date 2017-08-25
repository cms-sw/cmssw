#ifndef FWCore_SOA_Table_h
#define FWCore_SOA_Table_h
// -*- C++ -*-
//
// Package:     FWCore/SOA
// Class  :     Table
// 
/**\class Table Table.h "Table.h"

 Description: A Table which is a 'structure of arrays'

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 24 Aug 2017 16:18:05 GMT
//

// system include files
#include <tuple>
#include <array>

// user include files
#include "FWCore/SOA/interface/TableItr.h"
#include "FWCore/SOA/interface/tablehelpers.h"
#include "FWCore/SOA/interface/ColumnFillers.h"
#include "FWCore/SOA/interface/ColumnValues.h"
#include "FWCore/SOA/interface/RowView.h"

// forward declarations

namespace edm {
namespace soa {
  
  template <typename... Args>
  class Table {
  public:
    static constexpr const unsigned int kNColumns = sizeof...(Args);
    using Layout = std::tuple<Args...>;
    using const_iterator = ConstTableItr<Args...>;
    using iterator = TableItr<Args...>;
    
    template <typename T, typename... CArgs>
    Table(T const& iContainer, CArgs... iArgs): m_size(iContainer.size()) {
      using CtrChoice = std::conditional_t<sizeof...(CArgs)==0,
      CtrFillerFromAOS,
      CtrFillerFromContainers>;
      m_size = CtrChoice::fill(m_values,iContainer,std::forward<CArgs>(iArgs)...);
    }
    
    template<typename T, typename... CArgs>
    Table(T const& iContainer, ColumnFillers<CArgs...> iFiller) {
      m_size = iContainer.size();
      CtrFillerFromAOS::fillUsingFiller(iFiller,m_values, iContainer);
    }
    
    Table( Table<Args...> const& iOther):m_size(iOther.m_size), m_values{{nullptr}} {
      copyFromToWithResize<0>(m_size,iOther.m_values,m_values,std::true_type{});
    }
    
    Table( Table<Args...>&& iOther):m_size(0), m_values{{nullptr}} {
      std::swap(m_size,iOther.m_size);
      std::swap(m_values,iOther.m_values);
    }
    
    Table() : m_size(0) {
    }
    
    ~Table() {
      dtr<0>(m_values, std::true_type{});
    }
    
    Table<Args...>& operator=(Table<Args...>&& iOther) {
      Table<Args...> cp(std::move(iOther));
      std::swap(m_size,cp.m_size);
      std::swap(m_values, cp.m_values);
      return *this;
    }
    Table<Args...>& operator=(Table<Args...> const& iOther) {
      return operator=( Table<Args...>(iOther));
    }
    
    unsigned int size() const {
      return m_size;
    }
    
    template<typename U>
    typename U::type const& get(size_t iRow) const {
      return *(static_cast<typename U::type const*>(columnAddress<U>())+iRow);
    }
    template<typename U>
    typename U::type & get(size_t iRow)  {
      return *(static_cast<typename U::type*>(columnAddress<U>())+iRow);
    }
    
    template<typename U>
    ColumnValues<typename U::type> column() const {
      return ColumnValues<typename U::type>{static_cast<typename U::type*>(columnAddress<U>()), m_size};
    }
    template<typename U>
    MutableColumnValues<typename U::type> column() {
      return MutableColumnValues<typename U::type>{static_cast<typename U::type*>(columnAddress<U>()), m_size};
    }
    
    const_iterator begin() const { return const_iterator{m_values}; }
    const_iterator end() const { return const_iterator{m_values,size()}; }

    iterator begin() { return iterator{m_values}; }
    iterator end() { return iterator{m_values,size()}; }

    
    template<typename U>
    void const * columnAddressWorkaround( U const*) const {
      return columnAddress<U>();
    }

    void const * columnAddressByIndex(unsigned int iIndex) const {
      return m_values[iIndex];
    }
    
  private:
    
    // Member data
    unsigned int m_size = 0;
    std::array<void *, sizeof...(Args)> m_values = {{nullptr}};
    
    template<typename U>
    void const* columnAddress() const {
      return m_values[impl::GetIndex<0,U,Layout>::index];
    }
    
    template<typename U>
    void * columnAddress() {
      return m_values[impl::GetIndex<0,U,Layout>::index];
    }

    //Recursive destructor handling
    template <int I>
    static void dtr(std::array<void*, sizeof...(Args)>& iArray, std::true_type) {
      using Type = typename std::tuple_element<I,Layout>::type::type;
      delete [] static_cast<Type*>(iArray[I]);
      dtr<I+1>(iArray,std::conditional_t<I+1<sizeof...(Args), std::true_type, std::false_type>{});
    }

    template <int I>
    static void dtr(std::array<void*, sizeof...(Args)>& iArray, std::false_type) {
    }
    
    //Construct the Table using a container per column
    struct CtrFillerFromContainers {
      template<typename T, typename... U>
      static size_t fill(std::array<void *, sizeof...(Args)>& oValues, T const& iContainer, U... iArgs) {
        static_assert( sizeof...(Args) == sizeof...(U)+1, "Wrong number of arguments passed to Table constructor");
        ctrFiller<0>(oValues,iContainer.size(), iContainer,std::forward<U>(iArgs)...);
        return iContainer.size();
      }
    private:
      template<int I, typename T, typename... U>
      static void ctrFiller(std::array<void *, sizeof...(Args)>& oValues, size_t iSize, T const& iContainer, U... iU) {
        assert(iContainer.size() == iSize);
        using Type = typename std::tuple_element<I,Layout>::type::type;
        Type  * temp = new Type [iSize];
        unsigned int index = 0;
        for( auto const& v: iContainer) {
          temp[index] = v;
          ++index;
        }
        oValues[I] = temp;
        
        ctrFiller<I+1>(oValues, iSize, std::forward<U>(iU)... );
      }
      
      template<int I>
      static void ctrFiller(std::array<void *, sizeof...(Args)>& , size_t  ) {}
      
    };
    
    //Construct the Table using one container with each entry representing a row
    struct CtrFillerFromAOS {
      template<typename T>
      static size_t fill(std::array<void *, sizeof...(Args)>& oValues, T const& iContainer) {
        presize<0>(oValues,iContainer.size(),std::true_type{});
        unsigned index=0;
        for(auto&& item: iContainer) {
          fillElement<0>(item,index,oValues,std::true_type{});
          ++index;
        }
        return iContainer.size();
      }
      
      template<typename T, typename F>
      static size_t fillUsingFiller(F& iFiller, std::array<void *, sizeof...(Args)>& oValues, T const& iContainer) {
        presize<0>(oValues,iContainer.size(),std::true_type{});
        unsigned index=0;
        for(auto&& item: iContainer) {
          fillElementUsingFiller<0>(iFiller, item,index,oValues,std::true_type{});
          ++index;
        }
        return iContainer.size();
      }
      

    private:
      template<int I>
      static void presize(std::array<void *, sizeof...(Args)>& oValues, size_t iSize, std::true_type) {
        using Layout = std::tuple<Args...>;
        using Type = typename std::tuple_element<I,Layout>::type::type;
        oValues[I] = new Type[iSize];
        presize<I+1>(oValues,iSize, std::conditional_t<I+1==sizeof...(Args),
                     std::false_type,
                     std::true_type>{});
      }
      template<int I>
      static void presize(std::array<void *, sizeof...(Args)>& oValues, size_t iSize, std::false_type) {}
      
      template<int I, typename E>
      static void fillElement(E const& iItem, size_t iIndex, std::array<void *, sizeof...(Args)>& oValues,  std::true_type) {
        using Layout = std::tuple<Args...>;
        using ColumnType = typename std::tuple_element<I,Layout>::type;
        using Type = typename ColumnType::type;
        Type* pElement = static_cast<Type*>(oValues[I])+iIndex;
        *pElement = value_for_column(iItem, static_cast<ColumnType*>(nullptr));
        fillElement<I+1>(iItem, iIndex, oValues, std::conditional_t<I+1==sizeof...(Args),
                         std::false_type,
                         std::true_type>{});
      }
      template<int I, typename E>
      static void fillElement(E const& iItem, size_t iIndex, std::array<void *, sizeof...(Args)>& oValues,  std::false_type) {}
      
      
      template<int I, typename E, typename F>
      static void fillElementUsingFiller(F& iFiller, E const& iItem, size_t iIndex, std::array<void *, sizeof...(Args)>& oValues,  std::true_type) {
        using Layout = std::tuple<Args...>;
        using ColumnType = typename std::tuple_element<I,Layout>::type;
        using Type = typename ColumnType::type;
        Type* pElement = static_cast<Type*>(oValues[I])+iIndex;
        *pElement = iFiller.value(iItem, static_cast<ColumnType*>(nullptr));
        fillElementUsingFiller<I+1>(iFiller,iItem, iIndex, oValues, std::conditional_t<I+1==sizeof...(Args),
                                    std::false_type,
                                    std::true_type>{});
      }
      template<int I, typename E, typename F>
      static void fillElementUsingFiller(F&, E const& , size_t , std::array<void *, sizeof...(Args)>& oValues,  std::false_type) {}
      
    };

    
    template<int I>
    static void copyFromToWithResize(size_t iNElements, std::array<void *, sizeof...(Args)> const& iFrom, std::array<void*, sizeof...(Args)>& oTo, std::true_type) {
      using Layout = std::tuple<Args...>;
      using Type = typename std::tuple_element<I,Layout>::type::type;
      Type* oldPtr = static_cast<Type*>(oTo[I]);
      Type* ptr = new Type[iNElements];
      oTo[I]=ptr;
      std::copy(static_cast<Type const*>(iFrom[I]), static_cast<Type const*>(iFrom[I])+iNElements, ptr);
      delete [] oldPtr;
      copyFromToWithResize<I+1>(iNElements, iFrom, oTo, std::conditional_t<I+1 == sizeof...(Args), std::false_type, std::true_type>{} );
    }
    template<int I>
    static void copyFromToWithResize(size_t, std::array<void *, sizeof...(Args)> const& , std::array<void*, sizeof...(Args)>&, std::false_type) {}
    
    template<int I>
    static void resizeFromTo(size_t iOldSize, size_t iNewSize, std::array<void *, sizeof...(Args)>& ioArray) {
      using Layout = std::tuple<Args...>;
      using Type = typename std::tuple_element<I,Layout>::type::type;
      Type* ptr = static_cast<Type*>(ioArray[I]);
      ptr = new Type[iNewSize];
      auto nToCopy = std::min(iOldSize,iNewSize);
      std::copy(static_cast<Type const*>(ioArray[I]), static_cast<Type const*>(ioArray[I])+nToCopy, ptr);
      resizeFromTo<I+1>(iOldSize, iNewSize, ioArray, std::conditional_t<I+1 == sizeof...(Args), std::false_type, std::true_type>{} );
      delete [] ioArray[I];
      ioArray[I]=ptr;
    }
    template<int I>
    static void resizeFromTo(size_t, size_t, std::array<void *, sizeof...(Args)>& , std::false_type) {}
    
  };
  
    
  /* Table Type Manipulation */
  template <typename T1, typename T2> struct AddColumns;
  template <typename... T1, typename... T2>
  struct AddColumns<Table<T1...>, std::tuple<T2...>> {
    using type = Table<T1...,T2...>;
  };
  
  template <typename T1, typename T2>
  using AddColumns_t = typename AddColumns<T1,T2>::type;
  
  
  template <typename LHS, typename E, typename RHS> struct RemoveColumnCheck;
  template <typename LHS, typename E, typename T, typename... U>
  struct RemoveColumnCheck<LHS, E, std::tuple<T,U...>> {
    using type =   typename std::conditional<std::is_same<E, T>::value,
    typename AddColumns<LHS,std::tuple<U...>>::type,
    typename RemoveColumnCheck<typename AddColumns<LHS,std::tuple<T>>::type, E, std::tuple<U...>>::type>::type;
  };
  
  template <typename LHS, typename E>
  struct RemoveColumnCheck<LHS, E, std::tuple<>> {
    using type = LHS;
  };
  
  template <typename TABLE, typename E>
  struct RemoveColumn {
    using type = typename RemoveColumnCheck<Table<>,E, typename TABLE::Layout>::type;
  };
  
  template <typename TABLE, typename E>
  using RemoveColumn_t = typename RemoveColumn<TABLE,E>::type;

}
}


#endif
