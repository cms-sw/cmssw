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

  //................................ Pointer values (cast Long_t)
  Long_t   fObjectTEcnaGui;
  Long_t   fObjectTEcnaHeader;
  Long_t   fObjectTEcnaHistos;
  Long_t   fObjectTEcnaNArrayD;
  Long_t   fObjectTEcnaNumbering;
  Long_t   fObjectTEcnaParCout;
  Long_t   fObjectTEcnaParEcal;
  Long_t   fObjectTEcnaParHistos;
  Long_t   fObjectTEcnaParPaths;
  Long_t   fObjectTEcnaRead;
  Long_t   fObjectTEcnaResultType;
  Long_t   fObjectTEcnaRootFile;
  Long_t   fObjectTEcnaRun;
  Long_t   fObjectTEcnaWrite;
 
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

  Bool_t  RegisterPointer(const TString&, const Long_t&);
  Long_t  GetPointerValue(const TString&);
  void    NumberCreateObjectMessage(const TString&, const Long_t&, const Int_t&);
  void    NumberReuseObjectMessage(const TString&, const Long_t&, const Int_t&);
  
ClassDef(TEcnaObject,1)// Parameter management for CNA (Correlated Noises Analysis)

};

#endif   //    ZTR_TEcnaObject
