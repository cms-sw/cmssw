#ifndef MuScleFitEvent_h
#define MuScleFitEvent_h
 
#include "TObject.h"

class MuScleFitEvent : public TObject
{
 public:
   MuScleFitEvent() : 
     fRun(0), 
     fEvent(0), 
     fTrueNumPUvtx(0),
     fTrueNumInteractions(0),
     fNpv(0)
       {}
     
     MuScleFitEvent(const unsigned int initRun, const unsigned int initEvent, const int initNPUvtx, const float initTrueNI, const int initNpv) :            
       fRun(initRun),
       fEvent(initEvent),
       fTrueNumPUvtx(initNPUvtx),
       fTrueNumInteractions(initTrueNI),
       fNpv(initNpv)          
       {} 

    // Getters
       UInt_t run() const {return fRun;}
       UInt_t event() const {return fEvent;}
       Int_t nPUvtx() const {return fTrueNumPUvtx;}
       Float_t nTrueInteractions() const {return fTrueNumInteractions;}
       UInt_t npv() const {return fNpv;}


     UInt_t fRun;
     UInt_t fEvent;
     Int_t fTrueNumPUvtx;
     Float_t fTrueNumInteractions;
     UInt_t fNpv;

  
  ClassDef(MuScleFitEvent, 1)
    };

ClassImp(MuScleFitEvent)
  
#endif

