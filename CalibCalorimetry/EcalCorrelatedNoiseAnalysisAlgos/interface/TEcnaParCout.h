#ifndef ZTR_TEcnaParCout
#define ZTR_TEcnaParCout

#include <Riostream.h>

#include "TObject.h"
#include "TSystem.h"
#include "Riostream.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaObject.h"

///-----------------------------------------------------------
///   TEcnaParCout.h
///   Update: 16/02/2011
///   Author:    B.Fabbro (bernard.fabbro@cea.fr)
///              DSM/IRFU/SPP CEA-Saclay
///   Copyright: Those valid for CEA sofware
///
///   ECNA web page:
///     http://cms-fabbro.web.cern.ch/cms-fabbro/
///     cna_new/Correlated_Noise_Analysis/ECNA_cna_1.htm
///-----------------------------------------------------------
///

class TEcnaParCout : public TObject {

 private:

  //..... Attributes
  Int_t   fgMaxCar;   // Max nb of caracters for char*
  Int_t   fCnew,        fCdelete;
  TString fTTBELL;
  Int_t   fCnaCommand,  fCnaError;


 public:
  //..... Public attributes
  Int_t    fFlagPrint;
  Int_t    fCodePrintComments, fCodePrintWarnings, fCodePrintAllComments, fCodePrintNoComment;

  //..... Methods
           TEcnaParCout();
           TEcnaParCout(TEcnaObject*);
  virtual  ~TEcnaParCout();

  void     Init();
  Int_t    GetCodePrint(const TString&);

ClassDef(TEcnaParCout,1)// Parameter management for ECNA (Ecal Correlated Noises Analysis)
};

#endif   //    ZTR_TEcnaParCout
