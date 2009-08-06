
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <boost/ref.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/iterator/transform_iterator.hpp>

namespace cond {

 
  namespace ecalcond {
    
    typedef EcalChannelStatus Container;
    typedef Container::Items  Items;
    typedef Container::value_type  value_type;
    
    enum How { singleChannel, bySuperModule, barrel, endcap, all};
    
    int bad(Items const & cont) {
      return  std::count_if(cont.begin(),cont.end(),
			    boost::bind(std::greater<int>(),
					boost::bind(&value_type::getStatusCode,_1),0)
			    );
    }
    
    
    void extractBarrel(Container const & cont, std::vector<int> const &,  std::vector<float> & result) {
      result.resize(1);
      result[0] =  bad(cont.barrelItems());
    }
    void extractEndcap(Container const & cont, std::vector<int> const &,  std::vector<float> & result) {
      result.resize(1);
      result[0] = bad(cont.endcapItems());
    }
    void extractAll(Container const & cont, std::vector<int> const &,  std::vector<float> & result) {
      result.resize(1);
      result[0] = bad(cont.barrelItems())+bad(cont.endcapItems());
    }
    
    void extractSuperModules(Container const & cont, std::vector<int> const & which,  std::vector<float> & result) {
      // bho...
    }
    
    void extractSingleChannel(Container const & cont, std::vector<int> const & which,  std::vector<float> & result) {
      result.reserve(which.size());
      for (int i=0; i<which.size();i++) {
	result.push_back(cont[which[i]].getStatusCode());
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
  PayLoadInspector<EcalChannelStatus>::dump() const {
    std::stringstream ss;
    return ss.str();
    
  }
  
  template<>
  std::string PayLoadInspector<EcalChannelStatus>::summary() const {
    std::stringstream ss;
    ss << ecalcond::bad(object().barrelItems()) << ", " << ecalcond::bad(object().endcapItems());
    return ss.str();
  }
  

  template<>
  std::string PayLoadInspector<EcalChannelStatus>::plot(std::string const & filename,
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

PYTHON_WRAPPER(EcalChannelStatus,EcalChannelStatus);
