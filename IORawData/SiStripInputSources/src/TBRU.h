#ifndef ROOT_TBRU
#define ROOT_TBRU 

#include "TNamed.h"

using namespace std;
#define MAX_RU_SIZE (512*96*2)
class TBRU : public TNamed {
  
public:
  Int_t fInstance; // Instance 
  Int_t fBx; // Bunch crossing
  Int_t fSize; // Size of the buffer
  Int_t fBuffer[MAX_RU_SIZE]; // 
public:
  TBRU()
    {
      fInstance = -1; 
      fSize=0;
      
    }

  TBRU(Int_t inst)
    {
      fInstance = inst; 
      fSize= MAX_RU_SIZE;
    }
  ~TBRU() {;}

  inline void setInstance(Int_t i) {fInstance =i;}
  inline void setBunchCrossing(Int_t i) {fBx =i;}
  inline void setBufferSize(Int_t i) {fSize =i;}
  inline void copyBuffer(Int_t *i, Int_t s) {fSize=s;memcpy(fBuffer,i,s*sizeof(Int_t));}
  inline Int_t getInstance(){ return fInstance;}
  inline Int_t getBunchCrossing(){ return fBx;}
  inline Int_t getBufferSize(){ return fSize;}
  inline Int_t* getBuffer(){ return fBuffer;}
  ClassDef(TBRU,1) // Test Beam FED Raw Data
       };
#endif
