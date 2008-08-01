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
  class ValueExtractor<Pedestals>: public  BaseValueExtractor<Pedestals> {
  public:

    typedef Pedestals Class;
    typedef ExtractWhat<Class> What;
    static What what() { return What();}

    ValueExtractor(){}
    ValueExtractor(What const & what, std::vector<int> const& which)
      : m_which(which)
    {
      // here one can make stuff really complicated...
    }
    void compute(Class const & it){
      for (int i=0; i<m_which.size();i++) {
	if (m_which[i]<  it.m_pedestals.size())
	  add(it.m_pedestals[m_which[i]].m_mean);
      }
    }
  private:
    std::vector<int> m_which;
  };


  template<>
  std::string
  PayLoadInspector<Pedestals>::dump() const {
    Printer p;
    std::for_each(object->m_pedestals.begin(),
		  object->m_pedestals.end(),boost::bind(&Printer::doit,boost::ref(p),_1));
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
