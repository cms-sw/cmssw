#include "CondFormats/EcalObjects/interface/EcalPedestals.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <boost/ref.hpp>
#include <boost/bind.hpp>
#include <fstream>

namespace {
  struct Printer {
    void doit(EcalPedestal const & item) {
      for (int i=1; i<4; i++)
	ss << item.mean(i) << "," << item.rms(i) <<";";
      ss << " ";
    }
    std::stringstream ss;
  };
}

namespace cond {

  template<>
  class ValueExtractor<EcalPedestals>: public  BaseValueExtractor<EcalPedestals> {
  public:

    typedef EcalPedestals Class;
    ValueExtractor(){}
    ValueExtractor(std::string const & what, std::vector<int> const& which)
      : m_which(which)
    {
      // here one can make stuff really complicated... (select mean rms, 12,6,1)
      // ask to make average on selected channels...
    }
    void compute(Class const & it){
      for (int i=0; i<m_which.size();i++) {
	// absolutely arbitraty
	if (m_which[i]<  it.barrelItems().size())
	  add( it.barrelItems()[m_which[i]].mean_x12);
      }
    }
  private:
    std::vector<int> m_which;
  };


  template<>
  std::string
  PayLoadInspector<EcalPedestals>::dump() const {
    Printer p;
    std::for_each(object->barrelItems().begin(),object->barrelItems().end(),boost::bind(&Printer::doit,boost::ref(p),_1));
    p.ss <<"\n";
    std::for_each(object->endcapItems().begin(),object->endcapItems().end(),boost::bind(&Printer::doit,boost::ref(p),_1));
    p.ss << std::endl;
    return p.ss.str();
  }

   template<>
   std::string PayLoadInspector<EcalPedestals>::summary() const {
     std::stringstream ss;
     ss << "sizes="
	<< object->barrelItems().size() <<","
	<< object->endcapItems().size() <<";";
     ss << std::endl;
     return ss.str();
   }


  // return the real name of the file including extension...
  template<>
  std::string PayLoadInspector<EcalPedestals>::plot(std::string const & filename,
						   std::string const &, std::vector<int> const&, std::vector<float> const& ) const {
    std::string fname = filename + ".txt";
    std::ofstream f(fname.c_str());
    f << dump();
    return fname;
  }
  
}

PYTHON_WRAPPER(EcalPedestals,EcalPedestals);
