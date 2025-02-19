#include "CondFormats/RPCObjects/interface/RPCObPVSSmap.h"
#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

namespace cond {

  template<>
  class ValueExtractor<RPCObPVSSmap>: public  BaseValueExtractor<RPCObPVSSmap> {
  public:

    typedef RPCObPVSSmap Class;
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
  std::string PayLoadInspector<RPCObPVSSmap>::summary() const {
    std::stringstream ss;

    std::vector<RPCObPVSSmap::Item> const & vdetid = object().ObIDMap_rpc;


    for(unsigned int i = 0; i < vdetid.size(); ++i ){
      ss <<vdetid[i].dpid <<" "<<vdetid[i].region<<" "<<vdetid[i].ring<<" "<<vdetid[i].sector<<" "<<vdetid[i].station<<" "<<vdetid[i].layer<<" "<<vdetid[i].subsector<<" "<<vdetid[i].suptype<<" ";
    }

    return ss.str();
  }


  template<>
  std::string PayLoadInspector<RPCObPVSSmap>::plot(std::string const & filename,
                                                   std::string const &,
                                                   std::vector<int> const&,
                                                   std::vector<float> const& ) const {
    std::string fname = filename + ".png";
    std::ofstream f(fname.c_str());
    return fname;
  }


}


PYTHON_WRAPPER(RPCObPVSSmap,RPCObPVSSmap);

