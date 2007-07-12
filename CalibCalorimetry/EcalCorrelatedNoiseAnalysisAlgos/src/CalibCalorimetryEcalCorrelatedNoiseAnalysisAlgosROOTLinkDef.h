#include "TString.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TNArrayD.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TDistrib.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEBParameters.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEBNumbering.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaParameters.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaResultType.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaRootFile.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaHeaderEB.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaRunEB.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaReadEB.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaViewEB.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaDialogEB.h"


#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class TNArrayD+;
#pragma link C++ class TDistrib+;
#pragma link C++ class TEBParameters+;
#pragma link C++ class TEBNumbering+;
#pragma link C++ class TCnaParameters+;
#pragma link C++ class TCnaResultType+;
#pragma link C++ class TCnaRootFile+;
#pragma link C++ class TCnaHeaderEB+;
#pragma link C++ class TCnaRunEB+;
#pragma link C++ class TCnaReadEB+;
#pragma link C++ class TCnaViewEB+;
#pragma link C++ class TCnaDialogEB+;

#pragma link C++ global gCnaRootFile;

#endif
