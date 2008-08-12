buildfileTemplate = """
<library file="$_CLASS_NAME_-PyWrapper.cc" name=$_CLASS_NAME_-PyInterface>
<use name=CondCore/Utilities>
<use name=CondFormats/$_PACKAGE_>
<use name=boost>
<use name=boost_filesystem>
<use name=boost_python>
<use name=boost_regex>
<flags EDM_PLUGIN=1>
</library>
"""

wrapperTemplate = """
#include "CondFormats/$_PACKAGE_/interface/$_HEADER_FILE_.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

namespace cond {

  template<>
  class ValueExtractor<$_CLASS_NAME_>: public  BaseValueExtractor<$_CLASS_NAME_> {
  public:

    typedef $_CLASS_NAME_ Class;
    typedef ExtractWhat<Class> What;
    static What what() { return What();}

    ValueExtractor(){}
    ValueExtractor(What const & what)
    {
      // here one can make stuff really complicated...
    }
    void compute(Class const & it){
    }
  private:
  
  };


  template<>
  std::string
  PayLoadInspector<$_CLASS_NAME_>::dump() const {
    std::stringstream ss;
    return ss.str();
    
  }
  
  template<>
  std::string PayLoadInspector<$_CLASS_NAME_>::summary() const {
    std::stringstream ss;
    return ss.str();
  }
  

  template<>
  std::string PayLoadInspector<$_CLASS_NAME_>::plot(std::string const & filename,
						   std::string const &, 
						   std::vector<int> const&, 
						   std::vector<float> const& ) const {
    std::string fname = filename + ".png";
    std::ofstream f(fname.c_str());
    return fname;
  }


}

PYTHON_WRAPPER($_CLASS_NAME_,$_CLASS_NAME_);
"""
