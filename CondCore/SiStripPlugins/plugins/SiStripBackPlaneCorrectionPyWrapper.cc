
#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

namespace cond {

  template<>
  class ValueExtractor<SiStripBackPlaneCorrection>: public  BaseValueExtractor<SiStripBackPlaneCorrection> {
  public:

    typedef SiStripBackPlaneCorrection Class;
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


  //template<>
  //std::string PayLoadInspector<SiStripBackPlaneCorrection>::dump() const {
  ////object().dump();
  //return "HACKED111!!";
  //  
  //}
  
  template<>
  std::string PayLoadInspector<SiStripBackPlaneCorrection>::summary() const {
    std::stringstream ss;
    object().printSummary(ss);
    return ss.str();
  }
  

  template<>
  std::string PayLoadInspector<SiStripBackPlaneCorrection>::plot(std::string const & filename,
						   std::string const &, 
						   std::vector<int> const&, 
						   std::vector<float> const& ) const {
    std::string fname = filename + ".png";
    std::ofstream f(fname.c_str());
    return fname;
  }


}

PYTHON_WRAPPER(SiStripBackPlaneCorrection,SiStripBackPlaneCorrection);
