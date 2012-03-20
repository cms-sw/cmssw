#ifndef ZTR_TEcnaParCout
#define ZTR_TEcnaParCout

#include <Riostream.h>

#include "TObject.h"
#include "TSystem.h"
#include "Riostream.h"

//------------------------ TEcnaParCout.h -----------------
//
//   For questions or comments, please send e-mail to:
//
//   Bernard Fabbro             
//   fabbro@hep.saclay.cea.fr 
//--------------------------------------------------------

class TEcnaParCout : public TObject {

 private:

  //..... Attributes

  // static const Int_t fgMaxCar = 512;                   // <=== HYPER DANGEREUX !!!

  Int_t   fgMaxCar;   // Max nb of caracters for char*

  Int_t   fCnew,        fCdelete;

  TString fTTBELL;

  Int_t   fCnaCommand,  fCnaError;

  //...............................................................

 public:

  //..... Public attributes

  Int_t    fFlagPrint;
  Int_t    fCodePrintComments, fCodePrintWarnings, fCodePrintAllComments, fCodePrintNoComment;

  //..... Methods

           TEcnaParCout();
  virtual  ~TEcnaParCout();

  void     Init();

  Int_t    GetCodePrint(const TString);

ClassDef(TEcnaParCout,1)// Parameter management for CNA (Correlated Noises Analysis)

};

#endif   //    ZTR_TEcnaParCout
