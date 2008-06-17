#include "CondFormats/Calibration/interface/Pedestals.h"
#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <boost/ref.hpp>
#include <boost/bind.hpp>

namespace {
  struct Printer {
    void doit(Pedestals::Item const & item) {
      ss << item.m_mean << "," << item.m_variance <<"; ";
    }
    std::stringstream ss;
  };
}

namespace cond {

  template<>
  std::string
  PayLoadInspector<Pedestals>::print() const {
    Printer p;
    std::for_each(object->m_pedestals.begin(),object->m_pedestals.end(),boost::bind(&Printer::doit,boost::ref(p),_1));
    p.ss << std::endl;
    return p.ss.str();
  }

   template<>
   std::string PayLoadInspector<Pedestals>::summary() const {
     std::stringstream ss;
     ss << "size="<<object->m_pedestals.size() <<";";
     ss << std::endl;
     return ss.str();
   }
  
}

PYTHON_WRAPPER(Pedestals,Pedestal);
