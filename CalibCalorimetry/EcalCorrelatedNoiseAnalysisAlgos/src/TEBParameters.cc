//----------Author's Name:FX Gentit and B.Fabbro  DAPNIA/SPP CEN Saclay
//----------Copyright:Those valid for CEA software
//----------Modified:07/06/2007

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEBParameters.h"

ClassImp(TEBParameters)
//____________________________________________________________________________

TEBParameters::TEBParameters(){
// Constructor without argument. Call to Init()

  Init();
}

TEBParameters::~TEBParameters() {
//destructor

}

void TEBParameters::Init() {
//Set default values in all variables and init the counters fxxxCalc

  //--------------------------------- INIT parameters

  //....................... CMS/ECAL parameters

  fMaxSMInBarPlus  = (Int_t)18;                                // Maximum number of SuperModules in the barrel+
  fMaxSMInBarMinus = (Int_t)18;                                // Maximum number of SuperModules in the barrel- 
  fMaxSMInBarrel   = fMaxSMInBarPlus + fMaxSMInBarMinus;       // Maximum number of SuperModules in the barrel

  fMaxTowEtaInSM = (Int_t)17;                                  // Maximum number of towers in eta in a SuperModule
  fMaxTowPhiInSM = (Int_t)4;                                   // Maximum number of towers in phi in a SuperModule
  fMaxTowInSM    = (Int_t)fMaxTowEtaInSM*fMaxTowPhiInSM;       // Maximum number of towers in a SuperModule

  fMaxCrysEtaInTow = (Int_t)5;                                 // Maximum number of crystals in eta in a tower
  fMaxCrysPhiInTow = (Int_t)5;                                 // Maximum number of crystals in phi in a tower
  fMaxCrysInTow    = (Int_t)fMaxCrysEtaInTow*fMaxCrysPhiInTow; // Maximum number of crystals in a tower 

  fMaxCrysEtaInSM = (Int_t)fMaxTowEtaInSM*fMaxCrysEtaInTow;    // Maximum number of crystals in eta in a SuperModule
  fMaxCrysPhiInSM = (Int_t)fMaxTowPhiInSM*fMaxCrysPhiInTow;    // Maximum number of crystals in phi in a SuperModule
  fMaxCrysInSM    = (Int_t)fMaxTowInSM*fMaxCrysInTow;          // Maximum number of crystals in a SuperModule

  fMaxSampADC  = (Int_t)10;                                    // Maximum number of samples ADC

  fMaxEvtsInBurstPedRun = (Int_t)150;                          // Maximum number of events per burst in Pedestal Runs
}

//=========================================================================
//
//                 METHODS TO GET THE PARAMETERS
//
//=========================================================================


//----------------------------------------------- max SM in barrel
Int_t TEBParameters::MaxSMInBarPlus()
{
// Gives the maximum number of super-modules in the barrel+

  Int_t MaxSMInBarPlus = fMaxSMInBarPlus;
  return MaxSMInBarPlus;
}

Int_t TEBParameters::MaxSMInBarMinus()
{
// Gives the maximum number of super-modules in the barrel-

  Int_t MaxSMInBarMinus = fMaxSMInBarMinus;
  return MaxSMInBarMinus;
}

Int_t TEBParameters::MaxSMInBarrel()
{
// Gives the maximum number of super-modules in the barrel

  Int_t MaxSMInBarrel = fMaxSMInBarrel;
  return MaxSMInBarrel;
}


//----------------------------------------------- max tow in SM
Int_t TEBParameters::MaxTowEtaInSM()
{
// Gives the maximum number of towers in eta in a SuperModule

  Int_t MaxSMTowEta = fMaxTowEtaInSM;
  return MaxSMTowEta;
}

Int_t TEBParameters::MaxTowPhiInSM()
{
// Gives the maximum number of towers in phi in a SuperModule

  Int_t MaxSMTowPhi = fMaxTowPhiInSM;
  return MaxSMTowPhi;
}

Int_t TEBParameters::MaxTowInSM()
{
// Gives the maximum number of towers in a SuperModule

  Int_t MaxSMTow = fMaxTowInSM;
  return MaxSMTow;
}

//------------------------------------------------ Max Crys in tower
Int_t TEBParameters::MaxCrysEtaInTow()
{
// Gives the maximum  number of crystals in eta a tower

  Int_t MaxTowEchaEta = fMaxCrysEtaInTow;
  return MaxTowEchaEta;
}

Int_t TEBParameters::MaxCrysPhiInTow()
{
// Gives the maximum  number of crystals in phi in a tower

  Int_t MaxTowEchaPhi = fMaxCrysPhiInTow;
  return MaxTowEchaPhi;
}

Int_t TEBParameters::MaxCrysInTow()
{
// Gives the maximum  number of crystals in a tower

  Int_t MaxTowEcha = fMaxCrysInTow;
  return MaxTowEcha;
}
//---------------------------------------------- Max crys in SM
Int_t TEBParameters::MaxCrysEtaInSM()
{
// Gives the maximum  number of crystals in eta a SuperModule

  Int_t MaxSMEchaEta = fMaxCrysEtaInSM;
  return MaxSMEchaEta;
}

Int_t TEBParameters::MaxCrysPhiInSM()
{
// Gives the maximum  number of crystals in phi in a SuperModule

  Int_t MaxSMEchaPhi = fMaxCrysPhiInSM;
  return MaxSMEchaPhi;
}
//
Int_t TEBParameters::MaxCrysInSM()
{
// Gives the maximum  number of crystals in a SuperModule

  Int_t MaxSMEcha = fMaxCrysInSM;
  return MaxSMEcha;
}

//------------------------------------------ Max samp ADC
Int_t TEBParameters::MaxSampADC()
{
// Gives the maximum  number of samples ADC

  Int_t MaxSampADC = fMaxSampADC;
  return MaxSampADC;
}
