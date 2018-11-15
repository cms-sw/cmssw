#ifndef FWCore_SOA_ColumnFillers_h
#define FWCore_SOA_ColumnFillers_h
// -*- C++ -*-
//
// Package:     FWCore/SOA
// Class  :     ColumnFillers
// 
/**\class ColumnFillers ColumnFillers.h "ColumnFillers.h"

 Description: Controls how edm::soa::Table columns can be filled from C++ objects

 Usage:
    An edm::soa::Table<> can be filled from a container of objects if the appropriate overloaded
 functions are defined.
 
 The overloaded function must be called `value_for_column`. The function takes two arguments. The
 first argument is a const reference to the class in question, or its base class. The second argument
 is a pointer to the edm::soa::Column<> type. E.g.
 
 \code
     namespace reco {
        double value_for_column(Candidate const& iCand, edm::soa::Energy*) {
           return iCand.energy();
        }
     }
 \endcode

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 24 Aug 2017 22:22:42 GMT
//

// system include files

// user include files

// forward declarations
#include <tuple>

namespace edm {
namespace soa {

  template <typename... Args>
  class ColumnFillers {
    using Layout = std::tuple<Args...>;
    Layout m_fillers;
    
    template<int I, typename ELEMENT>
    decltype(auto) callFiller(ELEMENT&& iEl) {
      return std::get<I>(m_fillers).m_f(iEl);
    }
    
    template<int I, typename COLUMN, typename ELEMENT>
    typename COLUMN::type tryValue(ELEMENT&& iEl) {
      if constexpr( I < sizeof...(Args)) {
        using Pair = typename std::tuple_element<I,Layout>::type;
        using COL = typename Pair::Column_type;
        if constexpr(std::is_same<COL,COLUMN>::value) {
          return callFiller<I>(iEl);
        } else {
          //try another filler to see if it matches
          return tryValue<I+1,COLUMN>(iEl);
        }
      } else {
        //no matches so call overload function
        return value_for_column(iEl,static_cast<COLUMN*>(nullptr));
      }
    }
    
  public:
    ColumnFillers(Args... iArgs): m_fillers(std::forward<Args>(iArgs)...) {}
    
    template<typename ELEMENT, typename COLUMN>
    typename COLUMN::type value(ELEMENT&& iEl, COLUMN*) {
      return tryValue<0,COLUMN>(iEl);
    }
  };

  template<typename... Args>
  ColumnFillers<Args...> column_fillers(Args... iArgs) {
    return ColumnFillers<Args...>(std::forward<Args>(iArgs)...);
  }

}
}


#endif
