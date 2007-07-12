#ifndef ROOT_TCnaResultType
#define ROOT_TCnaResultType

#include "TNArrayD.h"

enum CnaResultTyp
  {cTypTowerNumbers,   cTypEv,              cTypVar,             cTypEvts,
   cTypEvtsXmin,       cTypEvtsXmax,
   cTypCovScc,         cTypCorScc,          cTypCovCss,          cTypCorCss,
   cTypEvCorCss,       cTypSigCorCss,
   cTypSvCorrecCovCss, cTypCovCorrecCovCss, cTypCorCorrecCovCss, cTypLastEvtNumber,
   cTypEvEv,           cTypEvSig,           cTypSigEv,           cTypSigSig,
   cTypSampTime,       cTypCovSccMos,       cTypCorSccMos,
   cTypCovMosccMot,    cTypCorMosccMot,     cTypEvtNbInLoop};

class TCnaResultType : public TObject {

protected:

public:

  CnaResultTyp   fTypOfCnaResult; //type of info in this class
  Int_t          fIthElement;     //Ith element in the entry of type fTypOfCnaResult
  Int_t          fUserChannel;    //Channel chosen by the user
  TNArrayD       fMatMat;         //1st matrix, used in case of MatMat
  TNArrayD       fMatHis;         //2nd matrix, used in case of MatHis

  TCnaResultType();

  TCnaResultType(CnaResultTyp, Int_t,
	      Int_t = 0,    Int_t = 0,
	      Int_t = 0,    Int_t = 0);

  ~TCnaResultType();
  void          SetSizeMat(Int_t, Int_t);
  void          SetSizeHis(Int_t, Int_t);

  CnaResultTyp  GetTypOfEntry(Int_t);

  ClassDef(TCnaResultType,1) //One leaf of the CNA root file
};
#endif
