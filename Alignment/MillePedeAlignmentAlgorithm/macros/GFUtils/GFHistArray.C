// Author: Gero Flucke <mailto:flucke@mail.desy.de>
//____________________________________
// GFHistArray
//   Author:      Gero Flucke
//   Date:        May 31st, 2002
//   last update: $Date: 2009/01/20 20:21:39 $  
//   by:          $Author: flucke $
//

#include <TH1.h>

#include "GFHistArray.h"

ClassImp(GFHistArray)

GFHistArray::GFHistArray(Int_t initCapacity, Int_t lowerBound) 
  : TObjArray(initCapacity, lowerBound)
{
  
}

GFHistArray::~GFHistArray()
{

}
  
  //    TObject        **GetObjectRef(TObject *obj) const;
void GFHistArray::AddFirst(TObject *obj)
{
  if(this->CheckObjOK(obj)) this->TObjArray::AddFirst(obj);
}
void GFHistArray::AddLast(TObject *obj)
{
  if(this->CheckObjOK(obj)) this->TObjArray::AddLast(obj);
}

void GFHistArray::AddAll(const TCollection *collection)
{
  TIter i(collection);
  while(TObject* obj = i.Next()){
    if(this->CheckObjOK(obj)) this->Add(obj);
  }
}

void GFHistArray::AddAll(const GFHistArray *hists)
{
  this->TObjArray::AddAll(hists);
}

void GFHistArray::AddAt(TObject *obj, Int_t idx)
{
  if(this->CheckObjOK(obj)) this->TObjArray::AddAt(obj, idx);
}

void GFHistArray::AddAtAndExpand(TObject *obj, Int_t idx)
{
  if(this->CheckObjOK(obj)) this->TObjArray::AddAtAndExpand(obj, idx);
}

Int_t GFHistArray::AddAtFree(TObject *obj)
{
  if(this->CheckObjOK(obj)) return this->TObjArray::AddAtFree(obj);
  else return -1;
}

void GFHistArray::AddAfter(const TObject *after, TObject *obj)
{
  if(this->CheckObjOK(obj)) this->TObjArray::AddAfter(after, obj);
}

void GFHistArray::AddBefore(const TObject *after, TObject *obj)
{
  if(this->CheckObjOK(obj)) this->TObjArray::AddBefore(after, obj);
}

TH1 *GFHistArray::RemoveAt(Int_t idx)
{
  return static_cast<TH1*>(this->TObjArray::RemoveAt(idx));
}

TH1 *GFHistArray::Remove(TObject *obj)
{
  return static_cast<TH1*>(this->TObjArray::Remove(obj));
}


// inline?:
TH1 *GFHistArray::At(Int_t idx) const
{
  return static_cast<TH1*>(this->TObjArray::At(idx));
}

TH1 *GFHistArray::UncheckedAt(Int_t i) const 
{ 
  return static_cast<TH1*>(TObjArray::UncheckedAt(i)); 
}

TH1 *GFHistArray::Before(const TObject *obj) const
{
  return static_cast<TH1*>(this->TObjArray::Before(obj)); 
}

TH1 *GFHistArray::After(const TObject *obj) const
{
  return static_cast<TH1*>(this->TObjArray::After(obj)); 
}

TH1 *GFHistArray::First() const
{
  return static_cast<TH1*>(this->TObjArray::First()); 
}

TH1 *GFHistArray::Last() const
{
  return static_cast<TH1*>(this->TObjArray::Last()); 
}

// no inline:
TH1 *GFHistArray::operator[](Int_t i) const
{
  return static_cast<TH1*>(this->TObjArray::At(i)); 
}

// TH1 *&GFHistArray::operator[](Int_t i)
TObject *&GFHistArray::operator[](Int_t i)
{
// should not be used...
  return this->TObjArray::operator[](i); 
}


Bool_t GFHistArray::CheckObjOK (TObject * histObj)
{
  // accept NULL pointer:
  return (!histObj || histObj->InheritsFrom(TH1::Class()));
}
