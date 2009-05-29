#include "CondFormats/RunInfo/interface/LuminosityInfo.h"
#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>
#include <sstream>

namespace cond {
  
  //implement my helper Extractor class
  template<>
  struct ExtractWhat<lumi::LuminosityInfo>{
  };

  template<>
  class ValueExtractor<lumi::LuminosityInfo>: public  BaseValueExtractor<lumi::LuminosityInfo> {
  public:
    
    typedef ExtractWhat<lumi::LuminosityInfo > What;
    static What what() { 
      return What();
    }
    
    ValueExtractor(){}
    ValueExtractor(What const & what):m_what(what)
    {
      
    }
    void compute( lumi::LuminosityInfo const& payload){
      //for now, no fancy, we are interested only in lumi average
      this->add(payload.lumiAverage());
    }
  private:
    What m_what;
  };


  template<>
  std::string
  PayLoadInspector<lumi::LuminosityInfo>::dump() const {
    std::stringstream ss;
    object().print(ss);
    return ss.str();
  }
  
  template<>
  std::string 
  PayLoadInspector<lumi::LuminosityInfo>::summary() const {
    std::stringstream ss;
    object().print(ss);
    return ss.str();
  }
  

  template<>
  std::string PayLoadInspector<lumi::LuminosityInfo>::plot(std::string const & filename,std::string const &, std::vector<int> const&, std::vector<float> const& ) const {
    std::string fname = filename + ".png";
    std::ofstream f(fname.c_str());
    return fname;
  }
}

namespace condPython{
  //expose payload from c++ to python a class(type) What by implementing the defineWhat method of InspectorPythonWrapper. Code based on boost::python. 
  //helper classes extractor
  template<>
  void defineWhat<lumi::LuminosityInfo>(){
    typedef cond::ExtractWhat<lumi::LuminosityInfo> What;
    //boost::python binding expose methods in What
    class_<What>("What",init<>())
      ;
  }
}//ns condPython


//define C++ class , Plugin name
PYTHON_WRAPPER(lumi::LuminosityInfo,LuminosityInfo);
