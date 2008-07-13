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
  class ValueExtractor<Pedestals> {
  public:

    typedef Pedestals Class;
    ValueExtractor(){}
    ValueExtractor(std::string const & what, std::vector<int> const& which)
      : m_which(which)
    {
      // here one can make stuff really complicated...
    }
    std::vector<float> const & values() const { return m_values;}
    void compute(Class const & it){
      m_values.clear();
      for (int i=0; i<m_which.size();i++) {
	if (m_which[i]<  it.m_pedestals.size())
	  m_values.push_back(it.m_pedestals[m_which[i]].m_mean);
      }
    }
  private:
    std::vector<float> m_values;
    std::vector<int> m_which;
  };


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
