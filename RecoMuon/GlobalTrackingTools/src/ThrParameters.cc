#include "RecoMuon/GlobalTrackingTools/interface/ThrParameters.h"
#include "FWCore/Framework/interface/ESHandle.h"

using namespace edm;
using namespace std;

ThrParameters::ThrParameters(const EventSetup* eSetup)
{
  // Read the threshold DB
  ESHandle<DYTThrObject> dytThresholdsH;
  
  // This try catch is just temporary and
  // will be removed as soon as the DYTThrObject
  // record is included in a GT.
  // It is necessary here for testing
  try {eSetup->get<DYTThrObjectRcd>().get(dytThresholdsH);
    dytThresholds = dytThresholdsH.product();
    isValidThdDB_ = true;
  } catch(...) {
    isValidThdDB_ = false;
  }

  // Ape are always filled even they're null
  ESHandle<AlignmentErrorsExtended> dtAlignmentErrorsExtended;
  eSetup->get<DTAlignmentErrorExtendedRcd>().get( dtAlignmentErrorsExtended );
  for ( std::vector<AlignTransformErrorExtended>::const_iterator it = dtAlignmentErrorsExtended->m_alignError.begin();
	it != dtAlignmentErrorsExtended->m_alignError.end(); it++ ) {
    CLHEP::HepSymMatrix error = (*it).matrix();
    GlobalError glbErr(error[0][0], error[1][0], error[1][1], error[2][0], error[2][1], error[2][2]);
    DTChamberId DTid((*it).rawId());
    dtApeMap.insert( pair<DTChamberId, GlobalError> (DTid, glbErr) );
  }
  ESHandle<AlignmentErrorsExtended> cscAlignmentErrorsExtended;
  eSetup->get<CSCAlignmentErrorExtendedRcd>().get( cscAlignmentErrorsExtended );
  for ( std::vector<AlignTransformErrorExtended>::const_iterator it = cscAlignmentErrorsExtended->m_alignError.begin();
	it != cscAlignmentErrorsExtended->m_alignError.end(); it++ ) {
    CLHEP::HepSymMatrix error = (*it).matrix();
    GlobalError glbErr(error[0][0], error[1][0], error[1][1], error[2][0], error[2][1], error[2][2]);
    CSCDetId CSCid((*it).rawId());
    cscApeMap.insert( pair<CSCDetId, GlobalError> (CSCid, glbErr) );
  }
}

ThrParameters::~ThrParameters() {}

