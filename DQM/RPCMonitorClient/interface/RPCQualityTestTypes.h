#ifndef RPCQualityTestTypes_H
#define RPCQualityTestTypes_H

#include <string>
namespace rpc_dqm {

  namespace qTestType{

    std::string  XRangeContent="XRangeContent";
    std::string  YRangeContent="YRangeContent";

    std::string  Comp2RefChi2="Comp2RefChi2";
    std::string  Comp2RefKolmogorov="Comp2RefKolmogorov";

    std::string  Comp2RefEqualsString="ContentsYRange";
    std::string  Comp2RefEqualInt="Comp2RefEqualInt";
    std::string  Comp2RefEqualFloat="Comp2RefEqualFloat";
    std::string  Comp2RefEqualH1="Comp2RefEqualH1";
    std::string  Comp2RefEqualH2="Comp2RefEqualH2";
    std::string  Comp2RefEqualH3="Comp2RefEqualH3";

 }

}


#endif
