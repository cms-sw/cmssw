
//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// TEcnaNArrayD  ROOT class for multidimensional arrays of Double_t         //
//                                                                          //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////
#ifndef ROOT_TEcnaNArrayD
#define ROOT_TEcnaNArrayD

#include "TObject.h"

class TEcnaNArrayD : public TObject {

protected:

  Int_t     fNd;  //dimension of the array
  Int_t     fN1;  //number of elements in the 1st dimension
  Int_t     fN2;  //number of elements in the 2nd dimension
  Int_t     fN3;  //number of elements in the 3rf dimension
  Int_t     fN4;  //number of elements in the 4th dimension
  Int_t     fN5;  //number of elements in the 5th dimension
  Int_t     fN6;  //number of elements in the 6th dimension
  Int_t     fNL;  //length of the array = fN1*fN2*fN3*fN4*fN5*fN6 + 1
  Double_t *fA;   //[fNL] Array of Double_t of dimension fNd

  void         Init();
  inline Int_t OneDim(Int_t) const;
  inline Int_t OneDim(Int_t,Int_t) const;
  inline Int_t OneDim(Int_t,Int_t,Int_t) const;
  inline Int_t OneDim(Int_t,Int_t,Int_t,Int_t) const;
  inline Int_t OneDim(Int_t,Int_t,Int_t,Int_t,Int_t) const;
  inline Int_t OneDim(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t) const;


public:

  TEcnaNArrayD();
  TEcnaNArrayD(const TEcnaNArrayD&);
  TEcnaNArrayD(Int_t);
  TEcnaNArrayD(Int_t,Int_t);
  TEcnaNArrayD(Int_t,Int_t,Int_t);
  TEcnaNArrayD(Int_t,Int_t,Int_t,Int_t);
  TEcnaNArrayD(Int_t,Int_t,Int_t,Int_t,Int_t);
  TEcnaNArrayD(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t);
  virtual ~TEcnaNArrayD();
  void     Clean();
  Double_t GetOverFlow() const { return fA[fNL-1]; }
  void     ReSet(Int_t);
  void     ReSet(Int_t,Int_t);
  void     ReSet(Int_t,Int_t,Int_t);
  void     ReSet(Int_t,Int_t,Int_t,Int_t);
  void     ReSet(Int_t,Int_t,Int_t,Int_t,Int_t);
  void     ReSet(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t);
  const Double_t &operator()(Int_t i1) const;
  const Double_t &operator()(Int_t i1,Int_t i2) const;
  const Double_t &operator()(Int_t i1,Int_t i2,Int_t i3) const;
  const Double_t &operator()(Int_t i1,Int_t i2,Int_t i3,Int_t i4) const;
  const Double_t &operator()(Int_t i1,Int_t i2,Int_t i3,Int_t i4,Int_t i5) const;
  const Double_t &operator()(Int_t i1,Int_t i2,Int_t i3,Int_t i4,Int_t i5,Int_t i6) const;
  Double_t &operator()(Int_t i1);
  Double_t &operator()(Int_t i1,Int_t i2);
  Double_t &operator()(Int_t i1,Int_t i2,Int_t i3);
  Double_t &operator()(Int_t i1,Int_t i2,Int_t i3,Int_t i4);
  Double_t &operator()(Int_t i1,Int_t i2,Int_t i3,Int_t i4,Int_t i5);
  Double_t &operator()(Int_t i1,Int_t i2,Int_t i3,Int_t i4,Int_t i5,Int_t i6);
  ClassDef(TEcnaNArrayD,1) //ROOT class for multidimensional arrays of Double_t
};
#endif
