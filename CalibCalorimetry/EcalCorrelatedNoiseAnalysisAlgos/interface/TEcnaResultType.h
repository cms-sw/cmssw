#ifndef ROOT_TEcnaResultType
#define ROOT_TEcnaResultType

#include "TMath.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaObject.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaNArrayD.h"

///-----------------------------------------------------------
///   TEcnaResultType.h
///   Update: 16/02/2011
///   Authors:   FX Gentit, B.Fabbro (bernard.fabbro@cea.fr)
///              DSM/IRFU/SPP CEA-Saclay
///   Copyright: Those valid for CEA sofware
///
///   ECNA web page:
///     http://cms-fabbro.web.cern.ch/cms-fabbro/
///     cna_new/Correlated_Noise_Analysis/ECNA_cna_1.htm
///-----------------------------------------------------------
///
///  Set the results types and dimensions 
///

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
  TEcnaResultType(TEcnaObject*);

  ~TEcnaResultType();
  void SetSizeMat(Int_t, Int_t);
  void SetSizeHis(Int_t, Int_t);

  CnaResultTyp GetTypOfEntry(Int_t);

  ClassDef(TEcnaResultType,1) //One leaf of the CNA root file
};
#endif
