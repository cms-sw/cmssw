#ifndef ZTR_TCnaParameters
#define ZTR_TCnaParameters

#include "TObject.h"
#include "TString.h"

#include "Riostream.h"

//------------------------ TCnaParameters.h -----------------
//
//   For questions or comments, please send e-mail to:
//
//   Bernard Fabbro             
//   fabbro@hep.saclay.cea.fr 
//--------------------------------------------------------

class TCnaParameters : public TObject {

 private:

  //..... Attributes

  // static const Int_t fgMaxCar = 512;                   // <=== HYPER DANGEREUX !!!

  Int_t   fgMaxCar;   // Max nb of caracters for char*

  Int_t   fCnew,        fCdelete;
  Int_t   fCnewRoot,    fCdeleteRoot;

  TString fTTBELL;

  TString fPeriodOfRun;     // Period of the run // A REVOIR POUR SUPPRESSION CAR EVOLUTIF
  TString fPeriod2002;
  TString fPeriod2003;
  TString fPeriod2004_1;
  TString fPeriod2004_2;
  TString fPeriod2004_3;
  TString fPeriod2004_4;
  TString fPeriod2004_5;
  TString fPeriod2004_6;
  TString fPeriod2005;
  TString fPeriod2006_1;
  TString fPeriod2006_2;

 public:
           TCnaParameters();
  virtual  ~TCnaParameters();

  void      Init();

  void      SetPeriodTitles();
  TString   PeriodOfRun(const Int_t&);
  Int_t     GetCodePrint(const TString);

ClassDef(TCnaParameters,1)// Dialog box + methods for CNA (Correlated Noises Analysis)

};

#endif   //    ZTR_TCnaParameter
