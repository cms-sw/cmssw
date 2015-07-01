#define G__DICTIONARY

#include "DataFormats/Common/interface/Wrapper.h"
#include "CMGTools/H2TauTau/interface/TriggerEfficiency.h"
#include "CMGTools/H2TauTau/interface/METSignificance.h"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"
#ifdef CMS_NOCXX11
#define SMATRIX_USE_COMPUTATION
#else
#define SMATRIX_USE_CONSTEXPR
#endif

#include <Math/SMatrix.h>

namespace {
  struct CMGTools_H2TauTau {

    TriggerEfficiency trigeff;
    cmg::METSignificance metsig_;
    edm::Wrapper<cmg::METSignificance> metsige_;
    std::vector<cmg::METSignificance> metsigv_;
    edm::Wrapper<std::vector<cmg::METSignificance> > metsigve_;


  };
}

namespace DataFormats_Math {
  struct dictionary {
    //Used by MET Significance matrix
    ROOT::Math::SMatrix<double,2> smat;
  };
}
