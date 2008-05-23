#include "CondFormats/Calibration/interface/Pedestals.h"
#include "CondCore/DBCommon/interface/TypedRef.h"

#include "CondCore/Utilities/interface/CondPyInterface.h"

#include "CondCore/IOVService/interface/IOVProxy.h"
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <boost/ref.hpp>
#include <boost/bind.hpp>
#include "CondCore/IOVService/interface/IOVProxy.h"

namespace {


  class PythonWrapper {
  public:
    typedef Pedestals Class;
    struct Printer {
      void doit(Class::Item const & item) {
	ss << item.m_mean << "," << item.m_variance <<"; ";
      }
      std::stringstream ss;
    };

    PythonWrapper(const cond::IOVElement & elem) : 
      object(*elem.db(),elem.payloadToken()){}

    std::string print() const {
      Printer p;
      std::for_each(object->m_pedestals.begin(),object->m_pedestals.end(),boost::bind(&Printer::doit,boost::ref(p),_1));
      p.ss << std::endl;
      return p.ss.str();
    }

    std::string summary() const;

    cond::TypedRef<Class> object;    

  };

}
