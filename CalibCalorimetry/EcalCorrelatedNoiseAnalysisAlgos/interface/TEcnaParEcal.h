#ifndef ROOT_TEcnaParEcal
#define ROOT_TEcnaParEcal

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaObject.h"

#include <Riostream.h>
#include "TNamed.h"

///-----------------------------------------------------------
///   TEcnaParEcal.h
///   Update: 06/04/2011
///   Author:    B.Fabbro (bernard.fabbro@cea.fr)
///              DSM/IRFU/SPP CEA-Saclay
///   Copyright: Those valid for CEA sofware
///
///   ECNA web page:
///     http://cms-fabbro.web.cern.ch/cms-fabbro/
///     cna_new/Correlated_Noise_Analysis/ECNA_cna_1.htm
///-----------------------------------------------------------
///
///   Init ECAL Parameter values
///

class TEcnaParEcal : public TNamed {

protected:

  void      Init();

private:

  Int_t   fgMaxCar;                    // Max nb of caracters for char* or TString
  TString fTTBELL;

  Int_t fCnew;          // flags for dynamical allocation
  Int_t fCdelete;       

  //....... Current subdetector flag and codes

  TString fFlagSubDet;
  TString fCodeEB;
  TString fCodeEE;

  //....... ECAL parameters

  //.......................... EB

  Int_t fMaxSampADCEB;           // Maximum number of samples ADC

  //  Int_t fMaxEvtsInBurstPedRunEB; // Maximum number of events per burst in Pedestal Runs

  Int_t fMaxSMEtaInEB;         // Maximum number of SMs in eta in EB
  Int_t fMaxSMPhiInEB;         // Maximum number of SMs in phi in EB

  Int_t fMaxSMInEBPlus;        // Maximum number of SMs in EB+
  Int_t fMaxSMInEBMinus;       // Maximum number of SMs in EB- 
  Int_t fMaxSMInEB;            // Maximum number of SMs in EB

  Int_t fMaxTowEtaInEB;        // Maximum number of towers in eta in EB
  Int_t fMaxTowPhiInEB;        // Maximum number of towers in phi in EB
  Int_t fMaxTowInEB;           // Maximum number of towers in EB

  Int_t fMaxTowEtaInSM;        // Maximum number of towers in eta in SM
  Int_t fMaxTowPhiInSM;        // Maximum number of towers in phi in SM
  Int_t fMaxTowInSM;           // Maximum number of towers in SM
                               // ( = fMaxTowEtaInSM*fMaxTowPhiInSM )

  Int_t fMaxCrysEtaInTow;      // Maximum number of crystals in eta in a tower
  Int_t fMaxCrysPhiInTow;      // Maximum number of crystals in phi in a tower
  Int_t fMaxCrysInTow;         // Maximum number of crystals in a tower
                               // ( = fMaxCrysEtaInTow*fMaxCrysPhiInTow )

  Int_t fMaxCrysEtaInSM;       // Maximum number of crystals in eta in SM
                               // ( = fMaxTowEtaInSM*fMaxCrysEtaInTow )

  Int_t fMaxCrysPhiInSM;       // Maximum number of crystals in phi in SM
                               // ( = fMaxTowPhiInSM*fMaxCrysPhiInTow )

  Int_t fMaxCrysInSM;          // Maximum number of crystals in SM
                               // ( = fMaxTowInSM*fMaxCrysInTow )

  //.......................... EE

  Int_t fMaxSampADCEE;           // Maximum number of samples ADC

  //  Int_t fMaxEvtsInBurstPedRunEE; // Maximum number of events per burst in Pedestal Runs

  Int_t fMaxDeeIXInEE;         // Maximum number of Dees in IX in EE
  Int_t fMaxDeeIYInEE;         // Maximum number of Dees in IY in EE

  Int_t fMaxDeeInEEPlus;       // Maximum number of Dees in EE+
  Int_t fMaxDeeInEEMinus;      // Maximum number of Dees in EE- 
  Int_t fMaxDeeInEE;           // Maximum number of Dees in EE

  Int_t fMaxSCIXInEE;          // Maximum number of SC's in IX in EE
  Int_t fMaxSCIYInEE;          // Maximum number of SC's in IY in EE
  Int_t fMaxSCEcnaInEE;        // Maximum number of SC's in the EE matrix

  Int_t fMaxSCIXInDee;         // Maximum number of super-crystals in IX in Dee
  Int_t fMaxSCIYInDee;         // Maximum number of super-crystals in IY in Dee
  Int_t fMaxSCEcnaInDee;       // Maximum ECNA number of super-crystals in the Dee matrix
                                 // ( = fMaxSCIXInDee*fMaxSCIYInDee )
  Int_t fMaxSCForConsInDee;    // Maximum number of super-crystals for construction in Dee
  Int_t fMaxSCForConsInEE;     // Maximum number of super-crystals for construction in EE

  Int_t fMaxCrysIXInSC;        // Maximum number of crystals in IX in a super-crystal
  Int_t fMaxCrysIYInSC;        // Maximum number of crystals in IY in a super-crystal
  Int_t fMaxCrysInSC;          // Maximum number of crystals in a super-crystal
                               // ( = fMaxCrysIXInSC*fMaxCrysIYInSC )

  Int_t fMaxCrysIXInDee;       // Maximum number of crystals in IX in Dee
                                 // ( = fMaxSCIXInDee*fMaxCrysIXInSC )

  Int_t fMaxCrysIYInDee;       // Maximum number of crystals in IY in Dee
                                 // ( = fMaxSCIYInDee*fMaxCrysIYInSC )

  Int_t fMaxCrysEcnaInDee;     // Maximum number of crystals in Dee matrix
                               // ( = fMaxSCEcnaInDee*fMaxCrysInSC )
  Int_t fMaxCrysForConsInDee;  // Maximum number of crystals for construction in Dee
                               // ( = fMaxSCForConsInDee*fMaxCrysInSC )

  Int_t fEmptyChannelsForIncompleteSCInDee; // Total number of empty channels for the incomplete SCs
  Int_t fEmptyChannelsInDeeMatrixIncompleteSCIncluded; // Total number of empty channels in Dee "Ecna" matrix
                                                       // (incomplete SCs included)

  Int_t fMaxDSInEEPlus;       // Maximum number of Data Sectors in EE+
  Int_t fMaxDSInEEMinus;      // Maximum number of Data Sectors in EE-
  Int_t fMaxDSInEE;           // Maximum number of Data Sectors in EE

  Int_t fNumberOfNotConnectedSCs; // for SCs 182, 178, 207, etc...
  Int_t fNumberOfNotCompleteSCs;  // for SCs 161, 216, 224, etc...

  //.......................... Stas (current Subdetector)

  Int_t fMaxSampADC;           // Maximum number of samples ADC

  //  Int_t fMaxEvtsInBurstPedRun; // Maximum number of events per burst in Pedestal Runs

  Int_t fMaxStexHocoInStas;    // Maximum number of Stex's in Hoco in Stas
  Int_t fMaxStexVecoInStas;    // Maximum number of Stex's in Veco in Stas
  Int_t fMaxStexInStasPlus;    // Maximum number of Stex's in Stas+
  Int_t fMaxStexInStasMinus;   // Maximum number of Stex's in Stas- 
  Int_t fMaxStexInStas;        // Maximum number of Stex's in Stas

  Int_t fMaxStinHocoInStas;    // Maximum number of Stin's in Hoco in Stas
  Int_t fMaxStinVecoInStas;    // Maximum number of Stin's in Veco in Stas
  Int_t fMaxStinEcnaInStas;    // Maximum number of Stin's in Stas

  Int_t fMaxStinHocoInStex;    // Maximum number of Stin's in Hoco in a Stex
  Int_t fMaxStinVecoInStex;    // Maximum number of Stin's in Veco in a Stex
  Int_t fMaxStinEcnaInStex;    // Maximum number of Stin's in a Stex ("Ecna" Stex matrix for Dee)
                               // ( = fMaxStinHocoInStex*fMaxStinVecoInStex )

  Int_t fMaxCrysHocoInStin;    // Maximum number of crystals in Hoco in a Stin
  Int_t fMaxCrysVecoInStin;    // Maximum number of crystals in Veco in a Stin
  Int_t fMaxCrysInStin;        // Maximum number of crystals in a Stin
                               // ( = fMaxCrysHocoInStin*fMaxCrysVecoInStin )

  Int_t fMaxCrysHocoInStex;    // Maximum number of crystals in Hoco in a Stex
                               // ( = fMaxStinHocoInStex*fMaxCrysHocoInStin )

  Int_t fMaxCrysVecoInStex;    // Maximum number of crystals in Veco in a Stex
                               // ( = fMaxStinVecoInStex*fMaxCrysVecoInStin )

  Int_t fMaxCrysEcnaInStex;    // Maximum number of crystals in a ECNA matrix Stex
                               // ( = fMaxStinEcnaInStex*fMaxCrysInStin )

  Int_t fMaxStinInStex;        // EB: Maximum number of towers in SM (= fMaxStinEcnaInStex = fMaxTowInSM)
                               // EE: Maximum number of SC for Construction in Dee (= fMaxSCForConsInDee)

  Int_t fMaxCrysInStex;        // EB: Maximum number of crystals in SM (= fMaxCrysEcnaInStex = fMaxCrysInSM)
                               // EE: Maximum number of crystals for Construction in Dee (= fMaxCrysForConsInDee) 

  //  Int_t fMaxStinForConsInStas; // Maximum number of towers in EB
                               // or Maximum number of SC for construction in EE

  //------------------------------- methods

 public:

  TEcnaParEcal(); 
  TEcnaParEcal(const TString&); 
  TEcnaParEcal(TEcnaObject*, const TString&);
  ~TEcnaParEcal();

  void    SetEcalSubDetector(const TString&);
  TString GetEcalSubDetector();

  //............................. EB
  Int_t MaxSampADCEB();
  //  Int_t MaxEvtsInBurstPedRunEB();

  Int_t MaxSMEtaInEB();
  Int_t MaxSMPhiInEB();

  Int_t MaxSMInEBPlus();
  Int_t MaxSMInEBMinus();
  Int_t MaxSMInEB();

  Int_t MaxTowEtaInEB();
  Int_t MaxTowPhiInEB();
  Int_t MaxTowInEB();

  Int_t MaxTowEtaInSM();
  Int_t MaxTowPhiInSM();
  Int_t MaxTowInSM();

  Int_t MaxCrysEtaInTow();
  Int_t MaxCrysPhiInTow();
  Int_t MaxCrysInTow();

  Int_t MaxCrysEtaInSM();
  Int_t MaxCrysPhiInSM();
  Int_t MaxCrysInSM();

  //............................. EE
  Int_t MaxSampADCEE();
  //  Int_t MaxEvtsInBurstPedRunEE();

  Int_t MaxDeeIXInEE();
  Int_t MaxDeeIYInEE();

  Int_t MaxDeeInEEPlus();
  Int_t MaxDeeInEEMinus();
  Int_t MaxDeeInEE();

  Int_t MaxSCIXInEE();
  Int_t MaxSCIYInEE();
  Int_t MaxSCEcnaInEE();  // default for MaxSCInEE()
  Int_t MaxSCInEE();
  Int_t MaxSCForConsInEE();

  Int_t MaxSCIXInDee();
  Int_t MaxSCIYInDee();
  Int_t MaxSCEcnaInDee();  // default for MaxSCInDee()
  Int_t MaxSCInDee();
  Int_t MaxSCForConsInDee();

  Int_t MaxCrysIXInSC();
  Int_t MaxCrysIYInSC();
  Int_t MaxCrysInSC();

  Int_t MaxCrysIXInDee();
  Int_t MaxCrysIYInDee();
  Int_t MaxCrysEcnaInDee();  // default for MaxCrysInDee()
  Int_t MaxCrysInDee();
  Int_t MaxCrysForConsInDee();
  Int_t EmptyChannelsInDeeMatrixIncompleteSCIncluded();

  Int_t MaxDSInEE();

  Int_t NumberOfNotConnectedSCs();
  Int_t NumberOfNotCompleteSCs();

  //............................. Current subdetector (Stin-Stex-Stas)
  Int_t MaxSampADC();

  Int_t MaxStexHocoInStas();
  Int_t MaxStexVecoInStas();

  Int_t MaxStexInStasPlus();
  Int_t MaxStexInStasMinus();
  Int_t MaxStexInStas();

  Int_t MaxStinHocoInStas();
  Int_t MaxStinVecoInStas();
  Int_t MaxStinEcnaInStas();
  //  Int_t MaxStinForConsInStas();

  Int_t MaxStinHocoInStex();
  Int_t MaxStinVecoInStex();
  Int_t MaxStinEcnaInStex();
  Int_t MaxStinInStex();

  Int_t MaxCrysHocoInStex();
  Int_t MaxCrysVecoInStex();
  Int_t MaxCrysEcnaInStex();
  Int_t MaxCrysInStex();

  Int_t MaxCrysHocoInStin();
  Int_t MaxCrysVecoInStin();
  Int_t MaxCrysInStin();

  ClassDef(TEcnaParEcal,1)  //Init of ECAL parameters
};
#endif
