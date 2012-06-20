#ifndef ROOT_TEcnaResultType
#define ROOT_TEcnaResultType

#include "TMath.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaNArrayD.h"

enum CnaResultTyp
  {cTypNumbers,     cTypMSp,         cTypSSp,
   cTypAvTno,       cTypAvLfn,       cTypAvHfn,
   cTypHfCov,       cTypHfCor,       cTypCovCss,     cTypCorCss,
   cTypMeanCorss,   cTypSigCorss,
   cTypAvPed,       cTypAvMeanCorss, cTypAvSigCorss, cTypNbOfEvts,
   cTypPed,         cTypTno,         cTypLfn,        cTypHfn,
   cTypAdcEvt,      cTypLfCov,       cTypLfCor,
   cTypLFccMoStins, cTypHFccMoStins, cTypEvtNbInLoop};   //  cTypEvtNbInLoop -> FREE

class TEcnaResultType : public TObject {

protected:

public:

  CnaResultTyp   fTypOfCnaResult; //type of info in this class
  Int_t          fIthElement;     //Ith element in the entry of type fTypOfCnaResult
  Int_t          fUserChannel;    //Channel chosen by the user
  TEcnaNArrayD   fMatMat;         //1st matrix, used in case of MatMat
  TEcnaNArrayD   fMatHis;         //2nd matrix, used in case of MatHis

  TEcnaResultType();
  
  // TEcnaResultType(CnaResultTyp, Int_t, Int_t = 0, Int_t = 0, Int_t = 0, Int_t = 0);
  
  ~TEcnaResultType();
  void SetSizeMat(Int_t, Int_t);
  void SetSizeHis(Int_t, Int_t);

  CnaResultTyp GetTypOfEntry(Int_t);

  ClassDef(TEcnaResultType,1) //One leaf of the CNA root file
};
#endif
