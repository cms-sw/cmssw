#include "CondFormats/Calibration/interface/Pedestals.h"
#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <boost/ref.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>

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

    struct DescrQuantity {
      std::vector<std::string> me;
      DescrQuantity() : me(2) {
	me[0]="mean";
	me[1]="variance";
      }
    };

    // example using multiple choice (not static to make the python binding easier
    /*static*/ std::vector<std::string> const & descr_quantity() {
      static DescrQuantity d;
      return d.me;
    }

    std::vector<int> m_which;
    std::vector<int> const & which() const { return m_which;}
    void set_which(std::vector<int> & i) { m_which.swap(i);}

    std::vector<int> m_quantity;
    std::vector<int> const & quantity() const { return m_quantity;}
    void set_quantity(std::vector<int> & i) { m_quantity.swap(i);}
  };

  template<>
  class ValueExtractor<Pedestals>: public  BaseValueExtractor<Pedestals> {
  public:

    typedef Pedestals Class;
    typedef ExtractWhat<Class> What;
    static What what() { return What();}

    typedef boost::function<float(Pedestals::Item const &)> Value;
    static std::vector<Value> const & allValues() {
      static std::vector<Value> v(2);
      // shall be done only ones in the usual constructor (later)...
      v[0]=boost::bind(&Pedestals::Item::m_mean,_1);
      v[1]=boost::bind(&Pedestals::Item::m_variance,_1);
      return v;
    }

    ValueExtractor(){}
    ValueExtractor(What const & what)
      : m_which(what.which()), m_what(what.quantity().size())
    {
      // here one can make stuff really complicated...
      std::vector<Value> const & v = allValues();
      for (size_t i=0; i< m_what.size(); i++) m_what[i] = v[what.quantity()[i]];
    }
    void compute(Class const & it){
      for (int i=0; i<m_which.size();i++) {
	if (m_which[i]<  it.m_pedestals.size())
	  for (size_t j=0; j< m_what.size(); j++) add(m_what[j](it.m_pedestals[m_which[i]]));
      }
    }
  private:
    std::vector<int> m_which;
    std::vector<Value> m_what;
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

  /* use default....
    template<>
    std::string PayLoadInspector<Pedestals>::summary() const {
    std::stringstream ss;
    ss << "size="<<object->m_pedestals.size() <<";";
    ss << std::endl;
    return ss.str();
   }
  */
}

namespace condPython {
  template<>
  void defineWhat<Pedestals>() {
    typedef cond::ExtractWhat<Pedestals> What;
    class_<What>("What",init<>())
      .def("set_which",&What::set_which)
      .def("which",&What::which, return_value_policy<copy_const_reference>())
      .def("set_quantity",&What::set_quantity)
      .def("quantity",&What::quantity, return_value_policy<copy_const_reference>())
      .def("descr_quantity",&What::descr_quantity, return_value_policy<copy_const_reference>())
      ;

  }
}

PYTHON_WRAPPER(Pedestals,Pedestal);
