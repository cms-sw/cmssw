#ifndef ZTR_Pattern
#define ZTR_Pattern
#include "TObject.h"

class TRawPattern : public TObject {

protected:

  Int_t  fNValue; //Number of values in fValues
  Int_t *fValues; //[fNValue] values of pattern unit

public:

  TRawPattern();
  TRawPattern(Int_t,Int_t*);
  virtual ~TRawPattern();

  Int_t    GetValue(Short_t j)  { return fValues[j]; }
  Int_t   *GetValues()          { return fValues;    }
  Int_t   GetNValues()          { return fNValue;   }

  Int_t   TestBit(Int_t unit, Short_t bit)
            { return ( fValues[unit] & (1<<bit) ); }
      
  virtual void     Print(const char *opt=0) const;
  void     SetValues( Int_t, Int_t * );
  virtual void     Clear(const char *opt="");
  void     Init();
  ClassDef(TRawPattern,1) //Content of the 3 pattern units used in the test beam
};
#endif
