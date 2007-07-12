//----------Author's Names:FX. Gentit, B.Fabbro  DAPNIA/SPP CEN Saclay
//----------Copyright:Those valid for CEA sofware
//----------Modified:07/06/2007

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaResultType.h"

#include "TMath.h"

ClassImp(TCnaResultType)
//___________________________________________________________________________
//
//  Set the results types and dimensions 
//
  
//.... Constructor without Arguments
  
TCnaResultType::TCnaResultType()
{
  fMatMat.ReSet(1,1);
  fMatHis.ReSet(1,1);
}

TCnaResultType::TCnaResultType(CnaResultTyp  typ,          Int_t   i,
			       Int_t         xArgNrowMat,  Int_t   xArgNcolMat,
			       Int_t         xArgNrowHis,  Int_t   xArgNcolHis)
{
//constructor

  fTypOfCnaResult = typ;
  fIthElement     = i;

  fMatMat.ReSet(1,1);
  fMatHis.ReSet(1,1);

  if ( (xArgNrowMat>0) && (xArgNcolMat>0) )
    {
      fMatMat.ReSet(xArgNrowMat,xArgNcolMat);
    }
  else
    {
      fMatMat.ReSet(1,1);
    }

  if ( (xArgNrowHis>0) && (xArgNcolHis>0) )
    {
      fMatHis.ReSet(xArgNrowHis,xArgNcolHis);
    }
  else 
    {
      fMatHis.ReSet(1,1);
    }
}

TCnaResultType::~TCnaResultType() {
//destructor
}

void TCnaResultType::SetSizeMat(Int_t nrow, Int_t ncol) {
//Set size of fMatMat
  nrow = TMath::Abs(nrow);
  ncol = TMath::Abs(ncol);
  if ((nrow>0) && (ncol>0)) fMatMat.ReSet(nrow,ncol);
  else                      fMatMat.ReSet(1,1);
}

void TCnaResultType::SetSizeHis(Int_t nrow, Int_t ncol) {
//Set size of fMatHis
  nrow = TMath::Abs(nrow);
  ncol = TMath::Abs(ncol);
  if ((nrow>0) && (ncol>0)) fMatHis.ReSet(nrow,ncol);
  else                      fMatHis.ReSet(1,1);
}

CnaResultTyp  TCnaResultType::GetTypOfEntry(Int_t  kEntry)
{
//Gives the type of the kth entry

//.......... List of the different types and associated parameters:
//
//    Nb of     Type of array            Type      Type
//   arrays                             Number 
//
//        1   fMatHis(1,tower)           ( 0)    cTypTowerNumbers  
//        1   fMatHis(channel, sample)   ( 1)    cTypEv    
//        1   fMatHis(channel, sample)   ( 2)    cTypVar
//  channel   fMatHis(sample,  bins)     ( 3)    cTypEvts,
//        1   fMatHis(channel, sample)   ( 4)    cTypEvtsXmin
//        1   fMatHis(channel, sample)   ( 5)    cTypEvtsXmax
//   sample   fMatMat(channel, channel)  ( 6)    cTypCovScc
//   sample   fMatMat(channel, channel)  ( 7)    cTypCorScc
//  channel   fMatMat(sample,  sample)   ( 8)    cTypCovCss
//  channel   fMatMat(sample,  sample)   ( 9)    cTypCorCss
//        1   fMatHis(1,channel)         (10)    cTypEvCorCss
//        1   fMatHis(1,channel)         (11)    cTypSigCorCss
//        1   fMatHis(channel, sample)   (12)    cTypSvCorrecCovCss
//  channel   fMatMat(sample,  sample)   (13)    cTypCovCorrecCovCss
//  channel   fMatMat(sample,  sample)   (14)    cTypCorCorrecCovCss
//        1   fMatHis(channel, sample)   (15)    cTypLastEvtNumber 
//        1   fMatHis(1,channel)         (16)    cTypEvEv
//        1   fMatHis(1,channel)         (17)    cTypEvSig
//        1   fMatHis(1,channel)         (18)    cTypSigEv
//        1   fMatHis(1,channel)         (19)    cTypSigSig
//  channel   fMatHis(sample,  bins)     (20)    cTypSampTime
//        1   fMatMat(channel, channel)  (21)    cTypCovSccMos
//        1   fMatMat(channel, channel)  (22)    cTypCorSccMos
//        1   fMatMat(channel, channel)  (23)    cTypCovMosccMot
//        1   fMatMat(channel, channel)  (24)    cTypCorMosccMot
//        1   fMatMat(tower, bin_evt)    (25)    cTypEvtNbInLoop     

  CnaResultTyp  xResultType = cTypTowerNumbers;   // (default)

 if (kEntry ==  0 ) { xResultType = cTypTowerNumbers;}
 if (kEntry ==  1 ) { xResultType = cTypEv;}
 if (kEntry ==  2 ) { xResultType = cTypVar;}
 if (kEntry ==  3 ) { xResultType = cTypEvts;}
 if (kEntry ==  4 ) { xResultType = cTypEvtsXmin;}
 if (kEntry ==  5 ) { xResultType = cTypEvtsXmax;}
 if (kEntry ==  6 ) { xResultType = cTypCovScc;}
 if (kEntry ==  7 ) { xResultType = cTypCorScc;}
 if (kEntry ==  8 ) { xResultType = cTypCovCss;}
 if (kEntry ==  9 ) { xResultType = cTypCorCss;}
 if (kEntry == 10 ) { xResultType = cTypEvCorCss;}
 if (kEntry == 11 ) { xResultType = cTypSigCorCss;}
 if (kEntry == 12 ) { xResultType = cTypSvCorrecCovCss;}
 if (kEntry == 13 ) { xResultType = cTypCovCorrecCovCss;}
 if (kEntry == 14 ) { xResultType = cTypCorCorrecCovCss;}
 if (kEntry == 15 ) { xResultType = cTypLastEvtNumber;}
 if (kEntry == 16 ) { xResultType = cTypEvEv;}
 if (kEntry == 17 ) { xResultType = cTypEvSig;}
 if (kEntry == 18 ) { xResultType = cTypSigEv;}
 if (kEntry == 19 ) { xResultType = cTypSigSig;}
 if (kEntry == 20 ) { xResultType = cTypSampTime;}
 if (kEntry == 21 ) { xResultType = cTypCovSccMos;}
 if (kEntry == 22 ) { xResultType = cTypCorSccMos;}
 if (kEntry == 23 ) { xResultType = cTypCovMosccMot;}
 if (kEntry == 24 ) { xResultType = cTypCorMosccMot;}
 if (kEntry == 25 ) { xResultType = cTypEvtNbInLoop;}

 return xResultType;
}
