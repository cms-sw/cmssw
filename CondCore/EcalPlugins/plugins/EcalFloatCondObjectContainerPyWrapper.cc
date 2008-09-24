
#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>


#include <sstream>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <boost/ref.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/iterator/transform_iterator.hpp>

namespace {
  struct Printer {
    Printer() : i(0){}
    void reset() { i=0;}
    void doB(float const & item) {
    }
    void doE(float const & item) {
      ss << i <<":"<< item << "\n";
      i++;
    }
    int i;
    std::stringstream ss;
  };
}


namespace cond {

  // migrate to a common trait (when fully understood)
  namespace ecalcond {

    typedef EcalFloatCondObjectContainer Container;
    typedef Container::value_type  value_type;

    enum How { singleChannel, bySuperModule, barrel, endcap, all};


    void extractBarrel(Container const & cont, std::vector<int> const &,  std::vector<float> & result) {
      result.resize(1);
      result[0] =  std::accumulate(cont.barrelItems().begin(),cont.barrelItems().end(),0.)/float(cont.barrelItems().size());
    }
    void extractEndcap(Container const & cont, std::vector<int> const &,  std::vector<float> & result) {
      result.resize(1);
      result[0] =  std::accumulate(cont.endcapItems().begin(),cont.endcapItems().end(),0.)/float(cont.endcapItems().size());
    }

     void extractAll(Container const & cont, std::vector<int> const &,  std::vector<float> & result) {
      result.resize(1);
      result[0] =  (std::accumulate(cont.barrelItems().begin(),cont.barrelItems().end(),0.)
		    +std::accumulate(cont.endcapItems().begin(),cont.endcapItems().end(),0.))
		    /float(cont.barrelItems().size()+cont.endcapItems().size());
    }
    
    void extractSuperModules(Container const & cont, std::vector<int> const & which,  std::vector<float> & result) {
      // bho...
    }

    void extractSingleChannel(Container const & cont, std::vector<int> const & which,  std::vector<float> & result) {
      result.reserve(which.size());
      for (int i=0; i<which.size();i++) {
	  result.push_back(cont[which[i]]);
      }
    }

    typedef boost::function<void(Container const & cont, std::vector<int> const & which,  std::vector<float> & result)> CondExtractor;

  } // ecalcond

  template<>
  struct ExtractWhat<ecalcond::Container> {
    
    ecalcond::How m_how;
    std::vector<int> m_which;
    
    ecalcond::How const & how() const { return m_how;}
    std::vector<int> const & which() const { return m_which;}
    
    void set_how(ecalcond::How i) {m_how=i;}
    void set_which(std::vector<int> & i) { m_which.swap(i);}
  };
  



  template<>
  class ValueExtractor<ecalcond::Container>: public  BaseValueExtractor<ecalcond::Container> {
  public:
    
    static ecalcond::CondExtractor & extractor(ecalcond::How how) {
      static  ecalcond::CondExtractor fun[5] = { 
	ecalcond::CondExtractor(ecalcond::extractSingleChannel),
	ecalcond::CondExtractor(ecalcond::extractSuperModules),
	ecalcond::CondExtractor(ecalcond::extractBarrel),
	ecalcond::CondExtractor(ecalcond::extractEndcap),
	ecalcond::CondExtractor(ecalcond::extractAll)
      };
      return fun[how];
    }
    
    
    typedef ecalcond::Container Class;
    typedef ExtractWhat<Class> What;
    static What what() { return What();}
    
    ValueExtractor(){}
    ValueExtractor(What const & what)
      : m_what(what)
    {
      // here one can make stuff really complicated... 
      // ask to make average on selected channels...
    }
    
    void compute(Class const & it){
      std::vector<float> res;
      extractor(m_what.how())(it,m_what.which(),res);
      swap(res);
    }
    
  private:
    What  m_what;
    
  };
  
  
  template<>
  std::string
  PayLoadInspector<EcalFloatCondObjectContainer>::dump() const {
    Printer p;
    std::for_each(object->barrelItems().begin(),object->barrelItems().end(),boost::bind(&Printer::doB,boost::ref(p),_1));
    p.ss <<"\n";
    p.reset();
    std::for_each(object->endcapItems().begin(),object->endcapItems().end(),boost::bind(&Printer::doE,boost::ref(p),_1));
    p.ss << std::endl;
    return p.ss.str();
  }
  
  template<>
  std::string PayLoadInspector<EcalFloatCondObjectContainer>::summary() const {
    std::stringstream ss;
    ss << "sizes="
       << object->barrelItems().size() <<","
       << object->endcapItems().size() <<";";
    ss << std::endl;
    return ss.str();
  }
  

  template<>
  std::string PayLoadInspector<EcalFloatCondObjectContainer>::plot(std::string const & filename,
								   std::string const &, 
								   std::vector<int> const&, 
								   std::vector<float> const& ) const {
    std::string fname = filename + ".png";
    std::ofstream f(fname.c_str());
    return fname;
  }
  
  
}

namespace condPython {
  template<>
  void defineWhat<cond::ecalcond::Container>() {
    enum_<cond::ecalcond::How>("How")
      .value("singleChannel",cond::ecalcond::singleChannel)
      .value("bySuperModule",cond::ecalcond::bySuperModule) 
      .value("barrel",cond::ecalcond::barrel)
      .value("endcap",cond::ecalcond::endcap)
      .value("all",cond::ecalcond::all)
      ;
    
    typedef cond::ExtractWhat<cond::ecalcond::Container> What;
    class_<What>("What",init<>())
      .def("set_how",&What::set_how)
      .def("set_which",&What::set_which)
      .def("how",&What::how, return_value_policy<copy_const_reference>())
      .def("which",&What::which, return_value_policy<copy_const_reference>())
      ;
  }
}


PYTHON_WRAPPER(EcalFloatCondObjectContainer,EcalFloatCondObjectContainer);
