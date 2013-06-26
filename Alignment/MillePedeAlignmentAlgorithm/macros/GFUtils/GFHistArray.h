#ifndef __GFHISTARRAY_H
#define __GFHISTARRAY_H

// ROOT includes
#include <TObjArray.h>
#include <TH1.h>

class TObject;

class GFHistArray : public TObjArray{
  // FIXME: friends needed?
friend class TObjArrayIter;
friend class TClonesArray;

public:
  explicit GFHistArray(Int_t s = TCollection::kInitCapacity, Int_t lowerBound = 0);
  virtual ~GFHistArray();
  
  //    TObject        **GetObjectRef(TObject *obj) const;
//   void              Add(TObject *obj) { AddLast(hist); }
  virtual void     AddFirst(TObject *obj);
  virtual void     AddLast(TObject *obj);
  virtual void     AddAll(const TCollection *collection);
  virtual void     AddAll(const GFHistArray *hists);
  virtual void     AddAt(TObject *obj, Int_t idx);
  virtual void     AddAtAndExpand(TObject *obj, Int_t idx);
  virtual Int_t     AddAtFree(TObject *obj);
  virtual void     AddAfter(const TObject *after, TObject *obj);
  virtual void     AddBefore(const TObject *before, TObject *obj);
  virtual TH1 *RemoveAt(Int_t idx);
  virtual TH1 *Remove(TObject *obj);

  TH1         *At(Int_t idx) const;
  TH1         *UncheckedAt(Int_t i) const;
  TH1         *Before(const TObject *obj) const;
  TH1         *After(const TObject *obj) const;
  TH1         *First() const;
  TH1         *Last() const;
  virtual TH1* operator[](Int_t i) const;
//   virtual TH1*&operator[](Int_t i); ??? warum nicht?

protected:
  Bool_t CheckObjOK (TObject * histObj);
 private:
  virtual TObject*& operator[](Int_t i); // invalidate const version, since cannot be overwritten

  ClassDef(GFHistArray,1) // type safe array of histograms
};
#endif // __GFHISTARRAY_H
