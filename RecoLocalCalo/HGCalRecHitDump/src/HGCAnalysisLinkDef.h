#include "RecoLocalCalo/HGCalRecHitDump/interface/HGCAnalysisTools.h"
#include "RecoLocalCalo/HGCalRecHitDump/interface/ROOTTools.h"
#include "RecoLocalCalo/HGCalRecHitDump/interface/PositionFit.h"


#ifdef __CINT__

#pragma link off all class; 
#pragma link off all function; 
#pragma link off all global; 
#pragma link off all typedef;

#pragma link C++ struct G4InteractionPositionInfo;
#pragma link C++ function getInteractionPosition;
#pragma link C++ function getEffSigma;
#pragma link C++ function getLambdaForHGCLayer;
#pragma link C++ struct CircleParams_t;
#pragma link C++ function fitCircleTo;
#pragma link C++ function circle_fcn;
#pragma link C++ enum _hgcpos_fitparams;
#pragma link C++ global _hgcpos_npts;
#pragma link C++ global _hgcpos_zff;
#pragma link C++ global _hgcpos_x;
#pragma link C++ global _hgcpos_y;
#pragma link C++ global _hgcpos_z;
#pragma link C++ global _hgcerr_x;
#pragma link C++ global _hgcerr_y;
#pragma link C++ function _hgcposfit_chiSquare;
#pragma link C++ function _hgcposfit_run;
#pragma link C++ function _hgcposfit_init;
#pragma link C++ struct HGCPositionFitResult_t;

#endif

// Local Variables:
// mode: c++
// mode: sensitive
// c-basic-offset: 8
// End:

