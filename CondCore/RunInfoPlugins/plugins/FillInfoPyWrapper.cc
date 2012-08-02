
#include "CondFormats/RunInfo/interface/FillInfo.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

namespace cond {

  template<>
  class ValueExtractor<FillInfo>: public  BaseValueExtractor<FillInfo> {
  public:

    typedef FillInfo Class;
    typedef ExtractWhat<Class> What;
    static What what() { return What();}

    ValueExtractor(){}
    ValueExtractor( What const & what )
    {
      // here one can make stuff really complicated...
    }
    void compute( Class const & it ){
      this->add( it.fillNumber() );
      this->add( it.bunchesInBeam1() );
      this->add( it.bunchesInBeam2() );
      this->add( it.collidingBunches() );
      this->add( it.targetBunches() );
      this->add( it.crossingAngle() );
      this->add( it.betaStar() );
      this->add( it.intensityForBeam1() );
      this->add( it.intensityForBeam2() );
      this->add( it.energy() );
      this->add( it.createTime() );
      this->add( it.beginTime() );
      this->add( it.endTime() );
    }
  private:
  
  };
  
  template<>
  std::string PayLoadInspector<FillInfo>::summary() const {
    std::stringstream ss;
    ss << this->object();
    return ss.str();
  }
  
  template<>
  std::string PayLoadInspector<FillInfo>::plot(std::string const & filename,
					       std::string const &, 
					       std::vector<int> const&, 
					       std::vector<float> const& ) const {
    std::string fname = filename + ".png";
    std::ofstream f(fname.c_str());
    return fname;
  }
  
}

PYTHON_WRAPPER(FillInfo,FillInfo);
