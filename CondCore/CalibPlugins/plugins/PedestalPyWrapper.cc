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
  struct ExtractWhat<Pedestals> {

    std::vector<int> m_which;

    std::vector<int> const & which() const { return m_which;}
 
    void set_which(std::vector<int> & i) { m_which.swap(i);}
  };

  template<>
  class ValueExtractor<Pedestals>: public  BaseValueExtractor<Pedestals> {
  public:

    typedef Pedestals Class;
    typedef ExtractWhat<Class> What;
    static What what() { return What();}

    ValueExtractor(){}
    ValueExtractor(What const & what)
      : m_which(what.which())
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

namespace condPython {
  template<>
  void defineWhat<Pedestals>() {
    typedef cond::ExtractWhat<Pedestals> What;
    class_<What>("What",init<>())
      .def("set_which",&What::set_which)
      .def("which",&What::which, return_value_policy<copy_const_reference>())
      ;

  }
}

PYTHON_WRAPPER(Pedestals,Pedestal);
