#ifndef ZTR_TEcnaObject
#define ZTR_TEcnaObject

#include <Riostream.h>
#include "TObject.h"
#include "TSystem.h"

#include "TString.h"

///-----------------------------------------------------------
///   TEcnaObject.h
///   Update: 15/02/2011
///   Author:    B.Fabbro (bernard.fabbro@cea.fr)
///              DSM/IRFU/SPP CEA-Saclay
///   Copyright: Those valid for CEA sofware
///
///   ECNA web page:
///     http://cms-fabbro.web.cern.ch/cms-fabbro/
///     cna_new/Correlated_Noise_Analysis/ECNA_cna_1.htm
///-----------------------------------------------------------
///
///   ECNA object management
///

class TEcnaObject : public TObject {

 private:

  //..... Attributes

  Int_t   fgMaxCar;

  TString fTTBELL;

  //................................ Pointer values (cast Int_t)
  Int_t   fObjectTEcnaGui;
  Int_t   fObjectTEcnaHeader;
  Int_t   fObjectTEcnaHistos;
  Int_t   fObjectTEcnaNArrayD;
  Int_t   fObjectTEcnaNumbering;
  Int_t   fObjectTEcnaParCout;
  Int_t   fObjectTEcnaParEcal;
  Int_t   fObjectTEcnaParHistos;
  Int_t   fObjectTEcnaParPaths;
  Int_t   fObjectTEcnaRead;
  Int_t   fObjectTEcnaResultType;
  Int_t   fObjectTEcnaRootFile;
  Int_t   fObjectTEcnaRun;
  Int_t   fObjectTEcnaWrite;
 
  //................................ Object creation counter
  Int_t   fCounterCreateTEcnaGui;
  Int_t   fCounterCreateTEcnaHeader;
  Int_t   fCounterCreateTEcnaHistos;
  Int_t   fCounterCreateTEcnaNArrayD;
  Int_t   fCounterCreateTEcnaNumbering;
  Int_t   fCounterCreateTEcnaParCout;
  Int_t   fCounterCreateTEcnaParEcal;
  Int_t   fCounterCreateTEcnaParHistos;
  Int_t   fCounterCreateTEcnaParPaths;
  Int_t   fCounterCreateTEcnaRead;
  Int_t   fCounterCreateTEcnaResultType;
  Int_t   fCounterCreateTEcnaRootFile;
  Int_t   fCounterCreateTEcnaRun;
  Int_t   fCounterCreateTEcnaWrite;

  //................................ Object re-using counter
  Int_t   fCounterReusingTEcnaGui;
  Int_t   fCounterReusingTEcnaHeader;
  Int_t   fCounterReusingTEcnaHistos;
  Int_t   fCounterReusingTEcnaNArrayD;
  Int_t   fCounterReusingTEcnaNumbering;
  Int_t   fCounterReusingTEcnaParCout;
  Int_t   fCounterReusingTEcnaParEcal;
  Int_t   fCounterReusingTEcnaParHistos;
  Int_t   fCounterReusingTEcnaParPaths;
  Int_t   fCounterReusingTEcnaRead;
  Int_t   fCounterReusingTEcnaResultType;
  Int_t   fCounterReusingTEcnaRootFile;
  Int_t   fCounterReusingTEcnaRun;
  Int_t   fCounterReusingTEcnaWrite;

 public:

  //..... Methods

           TEcnaObject();
  virtual  ~TEcnaObject();

  void Init();

  Bool_t  RegisterPointer(const TString, const Int_t&);
  void    NumberCreateObjectMessage(const TString, const Int_t&, const Int_t&);
  void    NumberReuseObjectMessage(const TString, const Int_t&, const Int_t&);
  Int_t   GetPointerValue(const TString);
  
ClassDef(TEcnaObject,1)// Parameter management for CNA (Correlated Noises Analysis)

};

#endif   //    ZTR_TEcnaObject
