#ifndef ROOT_TEBNumbering
#define ROOT_TEBNumbering

#include "TString.h"
#include "TObject.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEBParameters.h"

class TEBNumbering : public TObject {

private:

  Int_t       fCnew;          // flags for dynamical allocation
  Int_t       fCdelete;       

  TString     fTTBELL;

  TEBParameters*  fEcal;

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

  //...........................................

  Int_t**     fT2dSMCrys;
  Int_t*      fT1dSMCrys;
  Int_t*      fT1dSMTow;
  Int_t*      fT1dTowEcha;

  TString     fCodeChNumberingLvrbBot;
  TString     fCodeChNumberingLvrbTop;

  Int_t       fFlagPrint;
  Int_t       fCodePrintComments, fCodePrintWarnings, fCodePrintAllComments, fCodePrintNoComment;

protected:

  void      Init();

public:

  //------------------------------- methods

  TEBNumbering();
  ~TEBNumbering();

  void  BuildCrysTable();   // correspondance crystal# in super-module <-> (tower number, channel# in tower)

  Int_t GetSMCrysFromSMTowAndTowEcha(const Int_t&, const Int_t&);
  Int_t GetSMCrysFromSMEcha(const Int_t&);
  Int_t GetTowEchaFromSMEcha(const Int_t&);
  Int_t GetSMTowFromSMEcha(const Int_t&);
  Int_t GetTowEchaFromSMCrys(const Int_t&);
  Int_t GetSMTowFromSMCrys(const Int_t&);

  Double_t  GetEta(const Int_t&, const Int_t&, const Int_t&);
  Double_t  GetEtaMin(const Int_t&, const Int_t&);
  Double_t  GetEtaMax(const Int_t&, const Int_t&);

  Double_t  GetIEtaMin(const Int_t&, const Int_t&);    // only for axis of TowerCrystalNumbering (+0.5 shift)
  Double_t  GetIEtaMax(const Int_t&, const Int_t&);    // only for axis of TowerCrystalNumbering (-0.5 shift)
  Double_t  GetIEtaMin(const Int_t&);                  // only for axis of SMTowerNumbering (+0.5 shift)
  Double_t  GetIEtaMax(const Int_t&);                  // only for axis of SMTowerNumbering (-0.5 shift)

  Double_t  GetPhi(const Int_t&, const Int_t&, const Int_t&);
  Double_t  GetPhiMin(const Int_t&, const Int_t&);
  Double_t  GetPhiMax(const Int_t&, const Int_t&);

  Double_t  GetJPhiMin(const Int_t&, const Int_t&);
  Double_t  GetJPhiMax(const Int_t&, const Int_t&);
  Double_t  GetJPhiMin(const Int_t&);
  Double_t  GetJPhiMax(const Int_t&);

  Double_t  GetPhiMin(const Int_t&);
  Double_t  GetPhiMax(const Int_t&);

  Double_t  GetSMCentralPhi(const Int_t&);

  TString   GetXDirection(const Int_t&);
  TString   GetYDirection(const Int_t&);
  TString   GetJYDirection(const Int_t&);


  TString   GetTowerLvrbType(const Int_t&);
  TString   GetSMHalfBarrel(const Int_t&);

  //............... Flags Print Comments/Debug

  void  PrintNoComment();   // (default) Set flags to forbid the printing of all the comments
                            // except ERRORS
  void  PrintWarnings();    // Set flags to authorize printing of some warnings
  void  PrintComments();    // Set flags to authorize printing of infos and some comments
                            // concerning initialisations
  void  PrintAllComments(); // Set flags to authorize printing of all the comments

  ClassDef(TEBNumbering,1)  //Channel Numbering for CNA
};
#endif
