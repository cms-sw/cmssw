#include "Alignment/LaserAlignment/interface/TsosVectorCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"
//#include "LASCommissioningData.h"
//#include "LASGlobalLoop.h"
#include "Alignment/LaserAlignment/interface/LASGlobalData.h"
#include "Alignment/LaserAlignment/interface/LASCoordinateSet.h"
#include "Alignment/LaserAlignment/interface/LASModuleProfile.h"
#include "TH1.h"
//#include "TDirectory.h"

namespace Alignment_LaserAlignment {
  struct dictionary {
    TsosVectorCollection tsosesColl;
    edm::Wrapper<TsosVectorCollection> tsosesWrappedColl;
    LASGlobalData<int> lint;
    LASGlobalData<float> lfloat;
    LASGlobalData<std::vector<float> > lvfloat;
    LASGlobalData<LASCoordinateSet> lCoordinateSet;
    LASGlobalData<LASModuleProfile> ModuleProfile;
    LASGlobalData<std::pair<float, float> > lpff;
    LASGlobalData<unsigned int> luint;
    LASGlobalData<std::string> lstring;
    LASGlobalData<TH1D*> lthid;
    //  LASGlobalData<TDirectory*> ltdir;
  };
}  // namespace Alignment_LaserAlignment
