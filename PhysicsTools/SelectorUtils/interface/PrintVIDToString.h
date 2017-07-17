#ifndef __PhysicsTools_SelectorUtils_PrintVIDToString_h__
#define __PhysicsTools_SelectorUtils_PrintVIDToString_h__

#include "DataFormats/Common/interface/Ptr.h"
#include "PhysicsTools/SelectorUtils/interface/VersionedSelector.h"

#include <sstream>
#include <string>

template<typename T>
struct PrintVIDToString{  
  std::string operator()(const VersionedSelector<edm::Ptr<T> >& select) {
    std::stringstream out;
    select.print(out);
    return out.str();
  }
};

#endif
