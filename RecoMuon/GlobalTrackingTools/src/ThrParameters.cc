#include "RecoMuon/GlobalTrackingTools/interface/ThrParameters.h"
#include "FWCore/Framework/interface/ESHandle.h"

using namespace edm;
using namespace std;

ThrParameters::ThrParameters(ESHandle<DYTThrObject> dytThresholdsH,
                             const AlignmentErrorsExtended& dtAlignmentErrors,
                             const AlignmentErrorsExtended& cscAlignmentErrors) {
  if (dytThresholdsH.isValid()) {
    dytThresholds = dytThresholdsH.product();
    isValidThdDB_ = true;
  } else {
    dytThresholds = nullptr;
    isValidThdDB_ = false;
  }

  // Ape are always filled even they're null
  for (std::vector<AlignTransformErrorExtended>::const_iterator it = dtAlignmentErrors.m_alignError.begin();
       it != dtAlignmentErrors.m_alignError.end();
       it++) {
    CLHEP::HepSymMatrix error = (*it).matrix();
    GlobalError glbErr(error[0][0], error[1][0], error[1][1], error[2][0], error[2][1], error[2][2]);
    DTChamberId DTid((*it).rawId());
    dtApeMap.insert(pair<DTChamberId, GlobalError>(DTid, glbErr));
  }
  for (std::vector<AlignTransformErrorExtended>::const_iterator it = cscAlignmentErrors.m_alignError.begin();
       it != cscAlignmentErrors.m_alignError.end();
       it++) {
    CLHEP::HepSymMatrix error = (*it).matrix();
    GlobalError glbErr(error[0][0], error[1][0], error[1][1], error[2][0], error[2][1], error[2][2]);
    CSCDetId CSCid((*it).rawId());
    cscApeMap.insert(pair<CSCDetId, GlobalError>(CSCid, glbErr));
  }
}

ThrParameters::~ThrParameters() {}
