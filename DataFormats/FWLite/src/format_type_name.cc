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
// $Id$
//

// system include files
#include <string>

// user include files
#include "DataFormats/FWLite/interface/format_type_name.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

static const std::string s_symbolToMangled[] = {"::", "_1",
                                               "_" , "_2",
                                               "," , "_3",
                                               " " , "_4",
                                               "&" , "_7",
                                               "*" , "_8",
                                               "<" , "_9",
                                               ">" , "_0"
                                            };
static const unsigned int s_symbolToMangledSize = sizeof(s_symbolToMangled)/sizeof(std::string);
namespace fwlite {

  ///given a C++ class name returned a mangled name 
  std::string format_type_to_mangled(const std::string& iType) {
     std::string returnValue;
     returnValue.append(static_cast<std::string::size_type>(iType.size()*2),' ');
     std::string::size_type fromIndex=0;
     std::string::size_type toIndex=0;
     unsigned int sIndex=0;
     for(;fromIndex<iType.size();++fromIndex) {
        bool foundMatch = false;
        for(sIndex=0;sIndex<s_symbolToMangledSize;) {
           const std::string& symbol = s_symbolToMangled[sIndex];
           if(iType.substr(fromIndex,symbol.size())==symbol) {
              foundMatch = true;
              break;
           }
           ++sIndex;
           ++sIndex;
        }
        if(!foundMatch) {
           returnValue[toIndex]=iType[fromIndex];
           ++toIndex;
        } else {
           const std::string& mangled=s_symbolToMangled[sIndex+1];
           returnValue.replace(toIndex,mangled.size(),mangled);
           toIndex += mangled.size();
           fromIndex += s_symbolToMangled[sIndex].size()-1;
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
     unsigned int sIndex=0;
     for(;fromIndex<iMangled.size();++fromIndex) {
        bool foundMatch = false;
        for(sIndex=0;sIndex<s_symbolToMangledSize;) {
           const std::string& mangled = s_symbolToMangled[sIndex+1];
           if(iMangled.substr(fromIndex,mangled.size())==mangled) {
              foundMatch = true;
              break;
           }
           ++sIndex;
           ++sIndex;
        }
        if(!foundMatch) {
           returnValue[toIndex]=iMangled[fromIndex];
           ++toIndex;
        } else {
           const std::string& symbol=s_symbolToMangled[sIndex];
           returnValue.replace(toIndex,symbol.size(),symbol);
           toIndex += symbol.size();
           fromIndex += s_symbolToMangled[sIndex+1].size()-1;
        }
     }
     returnValue.resize(toIndex);
     return returnValue;
  }

}
