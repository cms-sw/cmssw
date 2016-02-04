#ifndef CDFChunk_h_included
#define CDFChunk_h_included 1

#include "TObject.h"

class CDFChunk : public TObject {
public:
  CDFChunk() { fChunkLength=0; fChunk=new ULong64_t[1]; }
  CDFChunk(const char* name) { fChunkLength=0; fChunk=0; fChunkName=name; }
  void adoptBuffer(ULong64_t* buffer, Int_t length) { fChunk=buffer; fChunkLength=length; fHeaderSize=2; fTrailerSize=1; }
  void releaseBuffer() { fChunk=0; fChunkLength=0; }
  void setChunkName(const char* name) { fChunkName=name; }
  inline ULong64_t* getData() { return fChunk; }
  inline Int_t getDataLength() const { return fChunkLength; }
  inline int getSourceId() const { return ((fChunk[0]>>8)&0xFFF); }
 private:
  TString fChunkName;
  Int_t fHeaderSize;
  Int_t fTrailerSize;
  Int_t fChunkLength;
  ULong64_t* fChunk; // [fChunkLength]
  ClassDef(CDFChunk,1)
};
#endif // CDFChunk_h_included
