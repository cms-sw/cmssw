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
    A Table provides a 'structure of arrays' using a spreadsheet metaphor.
 The template arguments of a Table should be edm::soa::Column<> types which 
 declare the type of the column and a label.
 \code
 constexpr char kEta[] = "eta";
 using Eta = edm::soa::Column<double,kEta>;

 constexpr char kPhi[] = "phi";
 using Phi = edm::soa::Column<double,kPhi>;

 using SphereTable = edm::soa::Table<Eta,Phi>;
 \endcode

 The same declaration of a column should be shared by different tables
 in order to allow the functions to be reused across tables. [See TableView]
 
 Accessing data within a Table is done by specifying the column and row
 to be used. The column is identified by the edm::soa::Column<> type and 
 the row by an integer.
 
 \code
 SphereTable sphereTable{...};
 ...
 auto eta1 = sphereTable.get<Eta>(1);
 \endcode
 
 One can iterate over all rows of a Table and get the values of interest
 \code
 SphereTable sphereTable{...};
 ...
 for(auto const& row: sphereTable) {
    std::cout<<row.get<Eta>()<<std::endl;
 }
 \encode
 If only some of the columns are of interest, the optimizer of the compiler
 is very good at removing code associated with non-used columns and providing
 a highly optimized access to just the data of interest.
 
 On can also explicitly iterate over a single column of interest
 \code
 SphereTable sphereTable{...};
 ...
 for(auto eta: sphereTable.column<Eta>()) {
    std::cout<<eta<<std::endl;
 }
 \encode
 Usually the optimizer on the compiler is able to make iteration over the entire
 row and iterating over just one column compile down to exactly the same machine
 instructions.
 
 A Table can be constructed either by
 1) passing as many containers as their are columns
 \code
   std::array<double, 4> eta = {...};
   std::array<double, 4> phi = {...};
   SphereTable sphereTable{eta,phi};
 \endcode
 2) passing a single container of objects where the objects hold the value of interests
 and where appropriate 'value_for_column' functions are defined [See ColumnFillers.h]
 \code
   class Vector {
     ...
     double eta() const;
     double phi() const;
     ...
   };
 
   double value_for_column(Vector const& iV, Eta*) { return iV.eta(); }
   double value_for_column(Vector const& iV, Phi*) { return iV.phi(); }
   ...
 
   std::vector<Vector> vectors{...};
   ...
   SphereTable sphereTable{ vectors };
 \endcode
 
 Functions which operate over Tables should not take Tables are arguments.
 Instead, they should take an edm::soa::TableView<>. This will allow the function
 to operate on any Table that uses the edm::soa::Column<> type needed by the function.
 \code
    SphereTable sphericalAngles(edm::soa::TableView<X,Y,Z>);
 \endcode

 
 New Table declarations can be created based on existing Table declarations.
 E.g. say you want a new Table based on an existing Table but with an additional column.
 \code
   using ATable = edm::soa::Table<...>;
 
   using MyLabel = edm::soa::Column<...>;
   using MyATable = AddColumns_t<ATable, MyLabel>;
 \endcode
 
 It is also possible to declare a new Table by removing columns from an existing declaration
 \code
   using MyBTable = RemoveColumn_t<BTable, Phi>; //Phi is a previously defined Column
 \endcode
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
    
    void resize(unsigned int iNewSize) {
      if(m_size == iNewSize) { return;}
      resizeFromTo<0>(m_size,iNewSize,m_values,std::true_type{});
      if(m_size < iNewSize) {
        //initialize the extra values
        resetStartingAt<0>(m_size, iNewSize,m_values,std::true_type{});
      }
      m_size = iNewSize;
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
    
    RowView<Args...> row(size_t iRow) const {
      return *(begin()+iRow);
    }
    MutableRowView<Args...> row(size_t iRow)  {
      return *(begin()+iRow);
    }
    
    const_iterator begin() const { 
      std::array<void const*, sizeof...(Args)> t;
      for(size_t i = 0; i<size();++i) { t[i] = m_values[i]; }
      return const_iterator{t}; }
    const_iterator end() const { 
      std::array<void const*, sizeof...(Args)> t;
      for(size_t i = 0; i<size();++i) { t[i] = m_values[i]; }
      return const_iterator{t,size()}; }

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
    static void resizeFromTo(size_t iOldSize, size_t iNewSize, std::array<void *, sizeof...(Args)>& ioArray, std::true_type) {
      using Layout = std::tuple<Args...>;
      using Type = typename std::tuple_element<I,Layout>::type::type;
      Type* oldPtr = static_cast<Type*>(ioArray[I]);
      auto ptr = new Type[iNewSize];
      auto nToCopy = std::min(iOldSize,iNewSize);
      std::copy(static_cast<Type const*>(ioArray[I]), static_cast<Type const*>(ioArray[I])+nToCopy, ptr);
      resizeFromTo<I+1>(iOldSize, iNewSize, ioArray, std::conditional_t<I+1 == sizeof...(Args), std::false_type, std::true_type>{} );
      
      delete [] oldPtr;
      ioArray[I]=ptr;
    }
    template<int I>
    static void resizeFromTo(size_t, size_t, std::array<void *, sizeof...(Args)>& , std::false_type) {}
    
    template<int I>
    static void resetStartingAt(size_t iStartIndex, size_t iEndIndex,std::array<void *, sizeof...(Args)>& ioArray,std::true_type) {
      using Layout = std::tuple<Args...>;
      using Type = typename std::tuple_element<I,Layout>::type::type;
      auto ptr = static_cast<Type*>(ioArray[I]);
      auto temp = Type{};
      std::fill(ptr+iStartIndex, ptr+iEndIndex, temp);
      resetStartingAt<I+1>(iStartIndex, iEndIndex, ioArray,std::conditional_t<I+1 == sizeof...(Args), std::false_type, std::true_type>{} );
    }

    template<int I>
    static void resetStartingAt(size_t, size_t,std::array<void *, sizeof...(Args)>& ,std::false_type) {
    }

  };
  
    
  /* Table Type Manipulation */
  template <typename T1, typename T2> struct AddColumns;
  template <typename... T1, typename... T2>
  struct AddColumns<Table<T1...>, std::tuple<T2...>> {
    using type = Table<T1...,T2...>;
  };
  
  template <typename T1, typename T2>
  using AddColumns_t = typename AddColumns<T1,T2>::type;
  
  namespace impl {
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
  }
    
  template <typename TABLE, typename E>
  struct RemoveColumn {
    using type = typename impl::RemoveColumnCheck<Table<>,E, typename TABLE::Layout>::type;
  };
  
  template <typename TABLE, typename E>
  using RemoveColumn_t = typename RemoveColumn<TABLE,E>::type;

}
}


#endif
