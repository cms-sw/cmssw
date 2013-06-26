// -*- C++ -*-
//
// Package:     FWLite
// Class  :     format_type_name
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Thu Dec  3 16:52:50 CST 2009
//

// system include files
#include <string>
#include "boost/static_assert.hpp"

// user include files
#include "DataFormats/FWLite/interface/format_type_name.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

static const std::string s_symbolDemangled[] = {"::",
                                                 "_" ,
                                                 "," ,
                                                 " " ,
                                                 "&" ,
                                                 "*" ,
                                                 "<" ,
                                                 ">" 
                                               };
static const std::string s_symbolMangled[] = {"_1",
                                              "_2",
                                              "_3",
                                              "_4",
                                              "_7",
                                              "_8",
                                              "_9",
                                              "_0"
                                             };
static const unsigned int s_symbolToMangledSize = sizeof(s_symbolDemangled)/sizeof(std::string);

namespace fwlite {

  void staticAssert() {
    BOOST_STATIC_ASSERT(sizeof(s_symbolMangled) == sizeof(s_symbolDemangled));
  }

  ///given a C++ class name returned a mangled name 
  std::string format_type_to_mangled(const std::string& iType) {
     std::string returnValue;
     returnValue.append(static_cast<std::string::size_type>(iType.size()*2),' ');
     std::string::size_type fromIndex=0;
     std::string::size_type toIndex=0;
     size_t sIndex=0;
     for(;fromIndex<iType.size();++fromIndex) {
        bool foundMatch = false;
        for(sIndex=0;sIndex<s_symbolToMangledSize;) {
           const std::string& symbol = s_symbolDemangled[sIndex];
           if(iType.substr(fromIndex,symbol.size())==symbol) {
              foundMatch = true;
              break;
           }
           ++sIndex;
        }
        if(!foundMatch) {
           returnValue[toIndex]=iType[fromIndex];
           ++toIndex;
        } else {
           const std::string& mangled=s_symbolMangled[sIndex];
           returnValue.replace(toIndex,mangled.size(),mangled);
           toIndex += mangled.size();
           fromIndex += s_symbolDemangled[sIndex].size()-1;
        }
     }
     returnValue.resize(toIndex);
     return returnValue;
  }

  ///given a mangled name return the C++ class name
  std::string unformat_mangled_to_type(const std::string& iMangled) {
     std::string returnValue;
     returnValue.append(static_cast<std::string::size_type>(iMangled.size()*2),' ');
     std::string::size_type fromIndex=0;
     std::string::size_type toIndex=0;
     size_t sIndex=0;
     for(;fromIndex<iMangled.size();++fromIndex) {
        bool foundMatch = false;
        for(sIndex=0;sIndex<s_symbolToMangledSize;) {
           const std::string& mangled = s_symbolMangled[sIndex];
           if(iMangled.substr(fromIndex,mangled.size())==mangled) {
              foundMatch = true;
              break;
           }
           ++sIndex;
        }
        if(!foundMatch) {
           returnValue[toIndex]=iMangled[fromIndex];
           ++toIndex;
        } else {
           const std::string& symbol=s_symbolDemangled[sIndex];
           returnValue.replace(toIndex,symbol.size(),symbol);
           toIndex += symbol.size();
           fromIndex += s_symbolMangled[sIndex].size()-1;
        }
     }
     returnValue.resize(toIndex);
     return returnValue;
  }

}
