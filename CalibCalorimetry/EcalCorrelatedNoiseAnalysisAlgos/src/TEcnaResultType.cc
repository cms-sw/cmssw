//----------Author's Names:FX. Gentit, B.Fabbro  DSM/IRFU/SPP CEA-Saclay
//----------Copyright:Those valid for CEA sofware
//----------Modified: 17/03/2010

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaResultType.h"
#include "Riostream.h"

ClassImp(TEcnaResultType)
//___________________________________________________________________________
//
//  Set the results types and dimensions 
//
  
//.... Constructor without Arguments
  
TEcnaResultType::TEcnaResultType()
{
  
  // cout << "[Info Management] CLASS: TEcnaResultType.    CREATE OBJECT: this = " << this << endl;

  fMatMat.ReSet(1,1);
  fMatHis.ReSet(1,1);
}

#define CWAR
#ifndef CWAR

TEcnaResultType::TEcnaResultType(CnaResultTyp  typ,          Int_t   i,
			       Int_t         xArgNrowMat,  Int_t   xArgNcolMat,
			       Int_t         xArgNrowHis,  Int_t   xArgNcolHis)
{
//constructor

  // cout << "[Info Management] CLASS: TEcnaResultType.    CREATE OBJECT: this = " << this << endl;

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

#endif // CWAR

TEcnaResultType::~TEcnaResultType() {
//destructor

  // cout << "[Info Management] CLASS: TEcnaResultType.    DESTROY OBJECT: this = " << this << endl;
}

void TEcnaResultType::SetSizeMat(Int_t nrow, Int_t ncol) {
//Set size of fMatMat
  nrow = TMath::Abs(nrow);
  ncol = TMath::Abs(ncol);
  if ((nrow>0) && (ncol>0)) fMatMat.ReSet(nrow,ncol);
  else                      fMatMat.ReSet(1,1);
}

void TEcnaResultType::SetSizeHis(Int_t nrow, Int_t ncol) {
//Set size of fMatHis
  nrow = TMath::Abs(nrow);
  ncol = TMath::Abs(ncol);
  if ((nrow>0) && (ncol>0)) fMatHis.ReSet(nrow,ncol);
  else                      fMatHis.ReSet(1,1);
}

CnaResultTyp  TEcnaResultType::GetTypOfEntry(Int_t  kEntry)
{
//Gives the type of the kth entry

//.......... List of the different types and associated parameters:
//
//    Nb of     Type of array            Type      Type
//   arrays                             Number 
//
//        1   fMatHis(1,tower)           ( 0)    cTypNumbers  
//        1   fMatHis(channel, sample)   ( 1)    cTypMSp    
//        1   fMatHis(channel, sample)   ( 2)    cTypSSp
//  channel   fMatHis(sample,  bins)     ( 3)    cTypAvTno,
//        1   fMatHis(channel, sample)   ( 4)    cTypAvLfn
//        1   fMatHis(channel, sample)   ( 5)    cTypAvHfn
//   sample   fMatMat(channel, channel)  ( 6)    cTypHfCov
//   sample   fMatMat(channel, channel)  ( 7)    cTypHfCor
//  channel   fMatMat(sample,  sample)   ( 8)    cTypCovCss
//  channel   fMatMat(sample,  sample)   ( 9)    cTypCorCss
//        1   fMatHis(1,channel)         (10)    cTypMeanCorss
//        1   fMatHis(1,channel)         (11)    cTypSigCorss
//        1   fMatHis(channel, sample)   (12)    cTypAvPed
//  channel   fMatMat(sample,  sample)   (13)    cTypAvMeanCorss
//  channel   fMatMat(sample,  sample)   (14)    cTypAvSigCorss
//        1   fMatHis(channel, sample)   (15)    cTypNbOfEvts 
//        1   fMatHis(1,channel)         (16)    cTypPed
//        1   fMatHis(1,channel)         (17)    cTypTno
//        1   fMatHis(1,channel)         (18)    cTypLfn
//        1   fMatHis(1,channel)         (19)    cTypHfn
//  channel   fMatHis(sample,  bins)     (20)    cTypAdcEvt
//        1   fMatMat(channel, channel)  (21)    cTypLfCov
//        1   fMatMat(channel, channel)  (22)    cTypLfCor
//        1   fMatMat(channel, channel)  (23)    cTypLFccMoStins
//        1   fMatMat(channel, channel)  (24)    cTypHFccMoStins
//        1   fMatMat(tower, bin_evt)    (25)    cTypEvtNbInLoop     

  CnaResultTyp  xResultType = cTypNumbers;   // (default)

 if (kEntry ==  0 ) { xResultType = cTypNumbers;}
 if (kEntry ==  1 ) { xResultType = cTypMSp;}
 if (kEntry ==  2 ) { xResultType = cTypSSp;}

 if (kEntry ==  3 ) { xResultType = cTypAvTno;}
 if (kEntry ==  4 ) { xResultType = cTypAvLfn;}
 if (kEntry ==  5 ) { xResultType = cTypAvHfn;}

 if (kEntry ==  6 ) { xResultType = cTypHfCov;}
 if (kEntry ==  7 ) { xResultType = cTypHfCor;}

 if (kEntry ==  8 ) { xResultType = cTypCovCss;}
 if (kEntry ==  9 ) { xResultType = cTypCorCss;}

 if (kEntry == 10 ) { xResultType = cTypMeanCorss;}
 if (kEntry == 11 ) { xResultType = cTypSigCorss;}

 if (kEntry == 12 ) { xResultType = cTypAvPed;}
 if (kEntry == 13 ) { xResultType = cTypAvMeanCorss;}
 if (kEntry == 14 ) { xResultType = cTypAvSigCorss;}

 if (kEntry == 15 ) { xResultType = cTypNbOfEvts;}

 if (kEntry == 16 ) { xResultType = cTypPed;}
 if (kEntry == 17 ) { xResultType = cTypTno;}
 if (kEntry == 18 ) { xResultType = cTypLfn;}
 if (kEntry == 19 ) { xResultType = cTypHfn;}

 if (kEntry == 20 ) { xResultType = cTypAdcEvt;}

 if (kEntry == 21 ) { xResultType = cTypLfCov;}
 if (kEntry == 22 ) { xResultType = cTypLfCor;}

 if (kEntry == 23 ) { xResultType = cTypLFccMoStins;}
 if (kEntry == 24 ) { xResultType = cTypHFccMoStins;}

 if (kEntry == 25 ) { xResultType = cTypEvtNbInLoop;}      // (FREE)

 return xResultType;
}
