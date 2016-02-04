//----------Author's Name:FX Gentit and B.Fabbro  DSM/IRFU/SPP CEA-Saclay
//----------Copyright:Those valid for CEA software
//----------Modified:25/09/2009

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParEcal.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaNumbering.h"

ClassImp(TEcnaParEcal)
//____________________________________________________________________________

TEcnaParEcal::TEcnaParEcal(){
// Constructor without argument. Call to Init()

 // cout << "[Info Management] CLASS: TEcnaParEcal.   CREATE OBJECT: this = " << this << endl;

  Init();
}

TEcnaParEcal::TEcnaParEcal(const TString SubDet){
// Constructor with argument. Call to Init() and set the subdetector flag

 // cout << "[Info Management] CLASS: TEcnaParEcal.   CREATE OBJECT: this = " << this << endl;

  Init();
  SetEcalSubDetector(SubDet.Data());
}

TEcnaParEcal::~TEcnaParEcal() {
//destructor

 // cout << "[Info Management] CLASS: TEcnaParEcal.   DESTROY OBJECT: this = " << this << endl;
}

void TEcnaParEcal::Init()
{
//Set values of Ecal parameters

  //--------------------------------- INIT parameters

  fTTBELL = '\007';

  fCnew       = 0;
  fCdelete    = 0;

  //....................... CMS/ECAL parameters

  //.............. Basic parameters for EB

  fMaxSampADCEB           = (Int_t)10;  // Maximum number of samples ADC

  fMaxEvtsInBurstPedRunEB = (Int_t)150; // Maximum number of events per burst in Pedestal Runs

  fMaxSMEtaInEB           = (Int_t)2;   // Maximum number of SuperModules in eta in the EB
  fMaxSMPhiInEB           = (Int_t)18;  // Maximum number of SuperModules in phi in the EB

  fMaxSMInEBPlus          = (Int_t)18;  // Maximum number of SuperModules in the EB+
  fMaxSMInEBMinus         = (Int_t)18;  // Maximum number of SuperModules in the EB- 

  fMaxTowEtaInSM          = (Int_t)17;  // Maximum number of towers in eta in a SuperModule
  fMaxTowPhiInSM          = (Int_t)4;   // Maximum number of towers in phi in a SuperModule

  fMaxCrysEtaInTow        = (Int_t)5;   // Maximum number of crystals in eta in a tower
  fMaxCrysPhiInTow        = (Int_t)5;   // Maximum number of crystals in phi in a tower

  //.............. Derived parameters for EB

  fMaxTowEtaInEB = fMaxSMEtaInEB*fMaxTowEtaInSM;      // Maximum number of towers in eta in EB
  fMaxTowPhiInEB = fMaxSMPhiInEB*fMaxTowPhiInSM;      // Maximum number of towers in phi in EB

  fMaxSMInEB      = fMaxSMInEBPlus + fMaxSMInEBMinus;         // Maximum number of SuperModules in the Ecal
  fMaxTowInSM     = (Int_t)fMaxTowEtaInSM*fMaxTowPhiInSM;     // Maximum number of towers in a SuperModule
  fMaxCrysInTow   = (Int_t)fMaxCrysEtaInTow*fMaxCrysPhiInTow; // Maximum number of crystals in a tower 

  fMaxCrysEtaInSM = (Int_t)fMaxTowEtaInSM*fMaxCrysEtaInTow;   // Maximum number of crystals in eta in a SuperModule
  fMaxCrysPhiInSM = (Int_t)fMaxTowPhiInSM*fMaxCrysPhiInTow;   // Maximum number of crystals in phi in a SuperModule
  fMaxCrysInSM    = (Int_t)fMaxTowInSM*fMaxCrysInTow;         // Maximum number of crystals in a SuperModule

  //.............. Basic parameters for the EE

  fMaxSampADCEE           = (Int_t)10;  // Maximum number of samples ADC

  fMaxEvtsInBurstPedRunEE = (Int_t)50;  // Maximum number of events per burst in Pedestal Runs

  fMaxDeeIXInEE           = (Int_t)4;   // Maximum number of Dees in IX in EE
  fMaxDeeIYInEE           = (Int_t)1;   // Maximum number of Dees in IY in EE

  fMaxDeeInEEPlus         = (Int_t)2;   // Maximum number of Dees in EE+
  fMaxDeeInEEMinus        = (Int_t)2;   // Maximum number of Dees in EE-

  fMaxSCIXInDee           = (Int_t)10;  // Maximum number of super-crystals in IX in Dee
  fMaxSCIYInDee           = (Int_t)20;  // Maximum number of super-crystals in IY in Dee
  fMaxSCForConsInDee      = (Int_t)149; // Maximum number of super-crystals for construction in Dee

  fMaxCrysIXInSC          = (Int_t)5;   // Maximum number of crystals in IX in a super-crystal
  fMaxCrysIYInSC          = (Int_t)5;   // Maximum number of crystals in IY in a super-crystal

  fMaxDSInEEPlus          = (Int_t)9;  // Maximum number of Data Sectors in EE+
  fMaxDSInEEMinus         = (Int_t)9;  // Maximum number of Data Sectors in EE-

  fNumberOfNotConnectedSCs = (Int_t)7; // Number of not connected SC's (178, 182, 207, 33, 29, etc... see EE mapping)
  fNumberOfNotCompleteSCs  = (Int_t)4; // Number of not complete  SC's (161, 216, 224, 12, 67, etc... see EE mapping)


  //.............. Derived parameters for the EE

  fMaxSCIXInEE    = fMaxDeeIXInEE*fMaxSCIXInDee;        // Maximum number of SC's in IX in EE
  fMaxSCIYInEE    = fMaxDeeIYInEE*fMaxSCIYInDee;        // Maximum number of SC's in IY in EE

  fMaxDeeInEE     = fMaxDeeInEEPlus + fMaxDeeInEEMinus; // Maximum number of Dees in EE
  fMaxSCEcnaInDee = fMaxSCIXInDee*fMaxSCIYInDee;        // Maximum number of super-crystals in the Dee matrix
  fMaxCrysInSC    = fMaxCrysIXInSC*fMaxCrysIYInSC;      // Maximum number of crystals in a super-crystal 

  fMaxCrysIXInDee      = fMaxSCIXInDee*fMaxCrysIXInSC;    // Maximum number of crystals in IX in Dee
  fMaxCrysIYInDee      = fMaxSCIYInDee*fMaxCrysIYInSC;    // Maximum number of crystals in IY in Dee
  fMaxCrysEcnaInDee    = fMaxSCEcnaInDee*fMaxCrysInSC;    // Max nb of crystals in the Dee matrix
  fMaxCrysForConsInDee = fMaxSCForConsInDee*fMaxCrysInSC; // Max nb of crystals for construction in Dee

  fMaxDSInEE = fMaxDSInEEPlus + fMaxDSInEEMinus;          // Maximum number of Data Sectors in EE

  //.............................. Current subdetector (Stas) parameters set to zero

  fMaxSampADC           = 0;

  fMaxEvtsInBurstPedRun = 0;

  fMaxStexHocoInStas    = 0;
  fMaxStexVecoInStas    = 0;

  fMaxStexInStasPlus    = 0;
  fMaxStexInStasMinus   = 0;
  fMaxStexInStas        = 0;

  fMaxStinHocoInStas    = 0; 
  fMaxStinVecoInStas    = 0; 

  fMaxStinHocoInStex    = 0;
  fMaxStinVecoInStex    = 0;
  fMaxStinEcnaInStex    = 0;

  fMaxCrysHocoInStin    = 0;
  fMaxCrysVecoInStin    = 0;
  fMaxCrysInStin        = 0;

  fMaxCrysHocoInStex    = 0;
  fMaxCrysVecoInStex    = 0;
  fMaxCrysEcnaInStex    = 0;

  //  fMaxStinForConsInStas = 0;

  fMaxStinInStex        = 0;
  fMaxCrysInStex        = 0;

  //.............................. Set codes for the Subdetector Flag
  fgMaxCar     = (Int_t)512;

  Int_t MaxCar = fgMaxCar;
  fCodeEB.Resize(MaxCar);
  fCodeEB = "EB";
 
  MaxCar = fgMaxCar;
  fCodeEE.Resize(MaxCar);
  fCodeEE = "EE";

  MaxCar = fgMaxCar;
  fFlagSubDet.Resize(MaxCar);
  fFlagSubDet = "No info";

} // end of Init()

void TEcnaParEcal::SetEcalSubDetector(const TString SubDet){
//Set the current subdetector flag and the current subdetector "Stin-Stex-Stas" parameters according to fFlagSubDet

  if( SubDet != fCodeEB && SubDet != fCodeEE )
    {
      cout << "!TEcnaParEcal::SetEcalSubDetector(...)> " << SubDet
	   << " : unknown subdetector code (requested: EB or EE)" << fTTBELL << endl;
    }
  else
    {
      Int_t MaxCar = fgMaxCar;
      fFlagSubDet.Resize(MaxCar);
      fFlagSubDet = SubDet.Data();   // Set the SubDetector flag

      if( fFlagSubDet != fCodeEB && fFlagSubDet != fCodeEE )
	{
	  cout << "!TEcnaParEcal::SetEcalSubDetector(...)> fFlagSubDet = " << fFlagSubDet
	       << " : CODE PROBLEM, subdetector flag not initialized." << fTTBELL << endl;
	}

      if(fFlagSubDet == fCodeEB)
	{
	  fMaxSampADC           = fMaxSampADCEB;
	  fMaxEvtsInBurstPedRun = fMaxEvtsInBurstPedRunEB;
	  
	  fMaxStexHocoInStas    = fMaxSMEtaInEB;
	  fMaxStexVecoInStas    = fMaxSMPhiInEB;

	  fMaxStexInStasPlus    = fMaxSMInEBPlus;
	  fMaxStexInStasMinus   = fMaxSMInEBMinus;
	  fMaxStexInStas        = fMaxSMInEB;
	  
	  fMaxStinHocoInStas    = fMaxTowEtaInEB; 
	  fMaxStinVecoInStas    = fMaxTowPhiInEB;

	  fMaxStinHocoInStex    = fMaxTowEtaInSM;
	  fMaxStinVecoInStex    = fMaxTowPhiInSM;
	  fMaxStinEcnaInStex    = fMaxTowInSM;
	  
	  fMaxCrysHocoInStin    = fMaxCrysEtaInTow;
	  fMaxCrysVecoInStin    = fMaxCrysPhiInTow;
	  fMaxCrysInStin        = fMaxCrysInTow;
	  
	  fMaxCrysHocoInStex    = fMaxCrysEtaInSM;
	  fMaxCrysVecoInStex    = fMaxCrysPhiInSM;
	  fMaxCrysEcnaInStex    = fMaxCrysInSM;

	  fMaxStinInStex        = fMaxTowInSM;
	  fMaxCrysInStex        = fMaxCrysInSM;
	}

      if(fFlagSubDet == fCodeEE)
	{
	  fMaxSampADC           = fMaxSampADCEE;
	  fMaxEvtsInBurstPedRun = fMaxEvtsInBurstPedRunEE;
	  
	  fMaxStexHocoInStas    = fMaxDeeIXInEE;
	  fMaxStexVecoInStas    = fMaxDeeIYInEE;

	  fMaxStexInStasPlus    = fMaxDeeInEEPlus;
	  fMaxStexInStasMinus   = fMaxDeeInEEMinus;
	  fMaxStexInStas        = fMaxDeeInEE;
	  
	  fMaxStinHocoInStas    = fMaxSCIXInEE; 
	  fMaxStinVecoInStas    = fMaxSCIYInEE;
	  
	  fMaxStinHocoInStex    = fMaxSCIXInDee;
	  fMaxStinVecoInStex    = fMaxSCIYInDee;
	  fMaxStinEcnaInStex    = fMaxSCEcnaInDee;
	  
	  fMaxCrysHocoInStin    = fMaxCrysIXInSC;
	  fMaxCrysVecoInStin    = fMaxCrysIYInSC;
	  fMaxCrysInStin        = fMaxCrysInSC;
	  
	  fMaxCrysHocoInStex    = fMaxCrysIXInDee;
	  fMaxCrysVecoInStex    = fMaxCrysIYInDee;
	  fMaxCrysEcnaInStex    = fMaxCrysEcnaInDee;

	  fMaxStinInStex        = fMaxSCForConsInDee;
	  fMaxCrysInStex        = fMaxCrysForConsInDee;
	}
    }
} // end of SetEcalSubDetector(const TString SubDet)
//======================================================================================
//
//                 METHODS TO GET THE PARAMETERS
//
//======================================================================================
//................................................................. SUBDETECTOR FLAG
TString TEcnaParEcal::GetEcalSubDetector(){return fFlagSubDet;}
//................................................................. BARREL
//------------------------------------------- Max samp ADC
Int_t TEcnaParEcal::MaxSampADCEB()   {return fMaxSampADCEB;} // maximum  number of samples ADC for EB
//------------------------------------------- Max number of events in Ped runs (for each gain)
Int_t TEcnaParEcal::MaxEvtsInBurstPedRunEB(){return fMaxEvtsInBurstPedRunEB;}
//------------------------------------------- Max SM in barrel
Int_t TEcnaParEcal::MaxSMEtaInEB() {return fMaxSMEtaInEB;}  // maximum number of SMs in eta in EB
Int_t TEcnaParEcal::MaxSMPhiInEB() {return fMaxSMPhiInEB;}  // maximum number of SMs in phi in EB

Int_t TEcnaParEcal::MaxSMInEBPlus() {return fMaxSMInEBPlus;}  // maximum number of SMs in the EB+
Int_t TEcnaParEcal::MaxSMInEBMinus(){return fMaxSMInEBMinus;} // maximum number of SMs in the EB-
Int_t TEcnaParEcal::MaxSMInEB()     {return fMaxSMInEB;}      // maximum number of SMs in EB
//------------------------------------------- Max tow in EB
Int_t TEcnaParEcal::MaxTowEtaInEB(){return fMaxTowEtaInEB;}   // maximum number of towers in eta in EB
Int_t TEcnaParEcal::MaxTowPhiInEB(){return fMaxTowPhiInEB;}   // maximum number of towers in phi in EB
//------------------------------------------- Max tow in SM
Int_t TEcnaParEcal::MaxTowEtaInSM()  {return fMaxTowEtaInSM;}   // maximum number of towers in eta in SM
Int_t TEcnaParEcal::MaxTowPhiInSM()  {return fMaxTowPhiInSM;}   // maximum number of towers in phi in SM
Int_t TEcnaParEcal::MaxTowInSM()     {return fMaxTowInSM;}      // maximum number of towers in SM
//------------------------------------------- Max Crys in tower
Int_t TEcnaParEcal::MaxCrysEtaInTow(){return fMaxCrysEtaInTow;} // maximum  number of crystals in eta a tower
Int_t TEcnaParEcal::MaxCrysPhiInTow(){return fMaxCrysPhiInTow;} // maximum  number of crystals in phi in a tower
Int_t TEcnaParEcal::MaxCrysInTow()   {return fMaxCrysInTow;}    // maximum  number of crystals in a tower
//------------------------------------------- Max crys in SM
Int_t TEcnaParEcal::MaxCrysEtaInSM() {return fMaxCrysEtaInSM;}  // maximum  number of crystals in eta in SM
Int_t TEcnaParEcal::MaxCrysPhiInSM() {return fMaxCrysPhiInSM;}  // maximum  number of crystals in phi in SM
Int_t TEcnaParEcal::MaxCrysInSM()    {return fMaxCrysInSM;}     // maximum  number of crystals in SM

//................................................................. ENDCAP
//------------------------------------------- Max samp ADC
Int_t TEcnaParEcal::MaxSampADCEE(){return fMaxSampADCEE;}   // maximum number of samples ADC for EE
//------------------------------------------- Max number of events in Ped runs (for each gain)
Int_t TEcnaParEcal::MaxEvtsInBurstPedRunEE(){return fMaxEvtsInBurstPedRunEE;}
//------------------------------------------- Max Dee in Endcap
Int_t TEcnaParEcal::MaxDeeIXInEE() {return fMaxDeeIXInEE;}  // maximum number of dees in IX in EE
Int_t TEcnaParEcal::MaxDeeIYInEE() {return fMaxDeeIYInEE;}  // maximum number of dees in IY in EE

Int_t TEcnaParEcal::MaxDeeInEEPlus() {return fMaxDeeInEEPlus;}  // maximum number of dees in EE+
Int_t TEcnaParEcal::MaxDeeInEEMinus(){return fMaxDeeInEEMinus;} // maximum number of dees in EE-
Int_t TEcnaParEcal::MaxDeeInEE()     {return fMaxDeeInEE;}      // maximum number of dees in EE
//------------------------------------------- Max SC in EE
Int_t TEcnaParEcal::MaxSCIXInEE(){return fMaxSCIXInEE;}         // maximum number of SC's in eta in EE
Int_t TEcnaParEcal::MaxSCIYInEE(){return fMaxSCIYInEE;}         // maximum number of SC's in phi in EE
//------------------------------------------- Max SC in Dee
Int_t TEcnaParEcal::MaxSCIXInDee()     {return fMaxSCIXInDee;}       // maximum number of SCs in IX in Dee
Int_t TEcnaParEcal::MaxSCIYInDee()     {return fMaxSCIYInDee;}       // maximum number of SCs in IY in Dee
Int_t TEcnaParEcal::MaxSCEcnaInDee()   {return fMaxSCEcnaInDee;}     // maximum number of SCs in the Dee matrix
Int_t TEcnaParEcal::MaxSCForConsInDee(){return fMaxSCForConsInDee;}  // max nb of crystals for construction in Dee
//------------------------------------------- Max Crys in SC
Int_t TEcnaParEcal::MaxCrysIXInSC(){return fMaxCrysIXInSC;}   // maximum number of crystals in IX in a SC
Int_t TEcnaParEcal::MaxCrysIYInSC(){return fMaxCrysIYInSC;}   // maximum number of crystals in IY in a SC
Int_t TEcnaParEcal::MaxCrysInSC()  {return fMaxCrysInSC;}     // maximum number of crystals in a SC
//------------------------------------------- Max crys in Dee
Int_t TEcnaParEcal::MaxCrysIXInDee()     {return fMaxCrysIXInDee;}      // max nb of crystals in IX in Dee
Int_t TEcnaParEcal::MaxCrysIYInDee()     {return fMaxCrysIYInDee;}      // max nb of crystals in IY in Dee
Int_t TEcnaParEcal::MaxCrysEcnaInDee()   {return fMaxCrysEcnaInDee;}    // max nb of crystals in Dee matrix
Int_t TEcnaParEcal::MaxCrysForConsInDee(){return fMaxCrysForConsInDee;} // max nb of crystals for construction in Dee
//------------------------------------------- Max DS in EE
Int_t TEcnaParEcal::MaxDSInEE(){return fMaxDSInEE;}
//------------------------------------------- Not connected and not complete SCs
Int_t TEcnaParEcal::NumberOfNotConnectedSCs(){return fNumberOfNotConnectedSCs;}
Int_t TEcnaParEcal::NumberOfNotCompleteSCs() {return fNumberOfNotCompleteSCs;}
 
//................................................................. Stas (current Subdetector)
//------------------------------------------- Max samp ADC
Int_t TEcnaParEcal::MaxSampADC()          {return fMaxSampADC;}          // max number of samples ADC
//------------------------------------------- Max number of events in Ped runs (for each gain)
Int_t TEcnaParEcal::MaxEvtsInBurstPedRun(){return fMaxEvtsInBurstPedRun;}
//------------------------------------------- Max Stex in Stas
Int_t TEcnaParEcal::MaxStexHocoInStas()  {return fMaxStexHocoInStas;}   // max number of Stexs in Hoco in Stas+
Int_t TEcnaParEcal::MaxStexVecoInStas()  {return fMaxStexVecoInStas;}   // max number of Stexs in Veco in Stas+

Int_t TEcnaParEcal::MaxStexInStasPlus()  {return fMaxStexInStasPlus;}   // max number of Stexs in Stas+
Int_t TEcnaParEcal::MaxStexInStasMinus() {return fMaxStexInStasMinus;}  // max number of Stexs in Stas-
Int_t TEcnaParEcal::MaxStexInStas()      {return fMaxStexInStas;}       // max number of Stexs in Stas
//------------------------------------------- Max Stin in Stas
Int_t TEcnaParEcal::MaxStinHocoInStas()   {return fMaxStinHocoInStas;}    // maximum number of Stin's in Hoco in Stas
Int_t TEcnaParEcal::MaxStinVecoInStas()   {return fMaxStinVecoInStas;}    // maximum number of Stin's in Veco in Stas
//------------------------------------------- Max Stin in Stex
Int_t TEcnaParEcal::MaxStinHocoInStex(){return fMaxStinHocoInStex;} // max number of Stins in Hoco in a Stex
Int_t TEcnaParEcal::MaxStinVecoInStex(){return fMaxStinVecoInStex;} // max number of Stins in Veco in a Stex
Int_t TEcnaParEcal::MaxStinEcnaInStex(){return fMaxStinEcnaInStex;} // max number of Stins in "ECNA matrix" Stex
Int_t TEcnaParEcal::MaxStinInStex()    {return fMaxStinInStex;}     // max number of Stins in Stex

//------------------------------------------- Max Crys in Stin
Int_t TEcnaParEcal::MaxCrysHocoInStin(){return fMaxCrysHocoInStin;} // max number of crystals in Hoco in a Stin
Int_t TEcnaParEcal::MaxCrysVecoInStin(){return fMaxCrysVecoInStin;} // max number of crystals in Veco in a Stin
Int_t TEcnaParEcal::MaxCrysInStin()    {return fMaxCrysInStin;}     // max number of crystals in a Stin
//------------------------------------------- Max crys in Stex
Int_t TEcnaParEcal::MaxCrysHocoInStex(){return fMaxCrysHocoInStex;} // max number of crystals in Hoco in a Stex
Int_t TEcnaParEcal::MaxCrysVecoInStex(){return fMaxCrysVecoInStex;} // max number of crystals in Veco in a Stex
Int_t TEcnaParEcal::MaxCrysEcnaInStex(){return fMaxCrysEcnaInStex;} // max number of crystals in "ECNA matrix" Stex
Int_t TEcnaParEcal::MaxCrysInStex()    {return fMaxCrysInStex;}     // max number of crystals in Stex  

