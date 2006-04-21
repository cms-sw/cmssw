#ifndef ZTR_TRawHeader
#define ZTR_TRawHeader
#include "TObject.h"

/* inline int numbit(int w) */
/* { */
/*   int num = 0; */
/*   if ( w < 0 ) {        // */
/*     w &= 0x7FFFFFFF;    // CINT Error */
/*     num++;              // */
/*   } */
/*   do { num += ( w & 0x1 ); } while ( (w >>= 1) != 0 ); */
/*   return num; */
/* } */

/* inline int ievtype(int trig_mask) */
/* { */
/*   int m = trig_mask & 0x00FFFF01; // bits 0, 8..23 */
/*   if ( numbit(m) != 1 )  { return 0; } */
/*   if ( m == 1 ) return 1; // Physics triggers */
/*   for(int i=0;i<24;i++) { */
/*     if ( ( m & 0x1 ) == 1 ) { */
/*       return i; */
/*     } */
/*     m >>= 1; */
/*   } */
/*   return 0; */
/* } */

class TRawHeader : public TObject {
 public:

  Int_t fBurstNum;            //Burst number
  Int_t fEvtNum;              //Event number
  Int_t fDate;                //Date
  Int_t fTrigMask;            //Trigger mask
  Int_t fFPPAMode;            //gains,dark,temp (99=mode auto)
  Int_t fPNMode;              //gains 0,1
  Int_t fLightIntensityIndex; //Light intensity index
  Int_t fInBeamXtal;          //Crystal on the beam
  Int_t fThetaTableIndex;     //Index table Theta
  Int_t fPhiTableIndex;       //Index table phi
  
  void  Init();
  

  TRawHeader() ;
  virtual ~TRawHeader() {}
  Int_t  GetBurstNum()    const { return fBurstNum; }
  Int_t  GetDate()        const { return fDate; }
  Int_t  GetEvtNum()      const { return fEvtNum; }
  Int_t  GetFPPAMode()    const { return fFPPAMode; }
  Int_t  GetPNMode()      const { return fPNMode; }
  Int_t  GetLightIntensityIndex() const { return fLightIntensityIndex; }
  Int_t  GetTrigMask()    const { return fTrigMask; }
  //  Int_t  GetEventType()   const { return ievtype(fTrigMask); }
  Int_t  GetXtal()        const { return fInBeamXtal; }
  Int_t  GetThetaTableIndex() const { return fThetaTableIndex; }
  Int_t  GetPhiTableIndex()   const { return fPhiTableIndex; }
  virtual void   Print(const char *opt=0)          const;
  void   Set(Int_t iv[] );

  ClassDef(TRawHeader,3)  //Event Header
};
#endif
