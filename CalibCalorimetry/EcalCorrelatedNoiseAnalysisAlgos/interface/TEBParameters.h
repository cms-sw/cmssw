#ifndef ROOT_TEBParameters
#define ROOT_TEBParameters

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEBParameters   Init ECAL Parameters                               //
//                                                                      //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNamed.h"

class TEBParameters : public TNamed {

protected:

  void      Init();

public:

  //....... ECAL parameters

  Int_t   fMaxSMInBarPlus;       // Maximum number of SuperModules in the barrel+
  Int_t   fMaxSMInBarMinus;      // Maximum number of SuperModules in the barrel- 
  Int_t   fMaxSMInBarrel;        // Maximum number of SuperModules in the barrel

  Int_t   fMaxTowEtaInSM;        // Maximum number of towers in eta in a SuperModule
  Int_t   fMaxTowPhiInSM;        // Maximum number of towers in phi in a SuperModule
  Int_t   fMaxTowInSM;           // Maximum number of towers in a SuperModule
                                 // ( = fMaxTowEtaInSM*fMaxTowPhiInSM )

  Int_t   fMaxCrysEtaInTow;      // Maximum number of crystals in eta in a tower
  Int_t   fMaxCrysPhiInTow;      // Maximum number of crystals in phi in a tower
  Int_t   fMaxCrysInTow;         // Maximum number of crystals in a tower
                                 // ( = fMaxCrysEtaInTow*fMaxCrysPhiInTow )

  Int_t   fMaxCrysEtaInSM;       // Maximum number of crystals in eta in a SuperModule
                                 // ( = fMaxTowEtaInSM*fMaxCrysEtaInTow )

  Int_t   fMaxCrysPhiInSM;       // Maximum number of crystals in phi in a SuperModule
                                 // ( = fMaxTowPhiInSM*fMaxCrysPhiInTow )

  Int_t   fMaxCrysInSM;          // Maximum number of crystals in a SuperModule
                                 // ( = fMaxTowInSM*fMaxCrysInTow )

  Int_t   fMaxSampADC;           // Maximum number of samples ADC

  Int_t   fMaxEvtsInBurstPedRun; // Maximum number of events per burst in Pedestal Runs

  //------------------------------- methods

  TEBParameters(); 
  ~TEBParameters();

  Int_t MaxSMInBarPlus();
  Int_t MaxSMInBarMinus();
  Int_t MaxSMInBarrel();

  Int_t MaxTowEtaInSM();
  Int_t MaxTowPhiInSM();
  Int_t MaxTowInSM();

  Int_t MaxCrysEtaInTow();
  Int_t MaxCrysPhiInTow();
  Int_t MaxCrysInTow();

  Int_t MaxCrysEtaInSM();
  Int_t MaxCrysPhiInSM();
  Int_t MaxCrysInSM();

  Int_t MaxSampADC();

  ClassDef(TEBParameters,1)  //Init of ECAL parameters
};
#endif
