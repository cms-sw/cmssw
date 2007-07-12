#ifndef   CL_TDistrib_H
#define   CL_TDistrib_H

#include "TObject.h"
#include "TString.h"
#include "TVectorD.h"

//--------------------------- TDistrib.h ---------------------------------
// 
//   Creation: 01 Jul  2002
//   Update  : 03 Feb  2003
//
//   For questions or comments, please send e-mail to:
//
//   Bernard Fabbro             
//   fabbro@hep.saclay.cea.fr 
//
//------------------------------------------------------------------------

//------------------------------------------------------------------------
//
//               Declaration de la classe TDistrib 
//
//       Une "Distribution" (classe "TDistrib") est definie comme
//       un ensemble de N valeurs qui representent les valeurs
//       que prend une variable aleatoire au cours de N evenements.
//       Les attributs de la classe sont: un entier ("Int_t") qui doit
//       contenir le nombre N d'evenements et un pointeur sur reels
//       longs (Double_t *) qui doit contenir l'adresse du premier element
//       (element numero 0) d'un tableau qui contient les valeurs.
//
//-----------------------------------------------------------------------

class TDistrib: public TObject {
  
 private:
  
  //............ attributes
  
  Int_t      fCnew;
  Int_t      fCdelete;
  
  Int_t      fNbEvents;       //  number of events of the distribution
  Double_t*  fValues;         //  pointer to the first value
  
  void     fCopy(const TDistrib&);   // private copy

  //................ memo errors tag
  Int_t    fCodeErrMax;

  Int_t    fMemoErrEv;
  Int_t    fMemoErrVar;
  Int_t    fMemoErrSdev;
  Int_t    fMemoErrCovar;
 
 public:
  
  //................. constructors
  
  TDistrib();
  //  TDistrib(const TDistrib&);
  TDistrib&  operator=(const TDistrib&);
  TDistrib(const Int_t&, const  Double_t*);
  TDistrib(const Int_t&, const  TVectorD);

  //................. destructor
  
  virtual ~TDistrib();
  
  //................. methods
  
  void          Init();
  Bool_t        Resize(const Int_t&);
  Bool_t        Refill(const Int_t&, const  TVectorD);

  Int_t         NumberOfEvents();
  Double_t*     VariableValues();
  Double_t*     DeviationValues();

  Double_t      ExpectationValue();
  Double_t      VarianceValue();
  Double_t      StandardDeviation();
  Double_t      StandardDeviation(TString);     
  Double_t      Covariance(TDistrib&);
  void          HistoDistrib(const Int_t&, Double_t&, Double_t&, Double_t*,
			     Int_t&, Int_t&, Int_t&);
  
  ClassDef(TDistrib,1)// Distribution 1D of a random variable
    
    };  
    
#endif     //   CL_TDistrib_H











