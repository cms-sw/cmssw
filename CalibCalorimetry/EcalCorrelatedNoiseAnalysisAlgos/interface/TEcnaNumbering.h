#ifndef ROOT_TEcnaNumbering
#define ROOT_TEcnaNumbering

#include "TString.h"
#include "TObject.h"
#include "Riostream.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaObject.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParEcal.h"

///-----------------------------------------------------------
///   TEcnaNumbering.h
///   Update: 30/06/2011
///   Author:    B.Fabbro (bernard.fabbro@cea.fr)
///              DSM/IRFU/SPP CEA-Saclay
///   Copyright: Those valid for CEA sofware
///
///   ECNA web page:
///     http://cms-fabbro.web.cern.ch/cms-fabbro/
///     cna_new/Correlated_Noise_Analysis/ECNA_cna_1.htm
///-----------------------------------------------------------

class TEcnaNumbering : public TObject {

private:

  Int_t fgMaxCar;   // Max nb of caracters for char*
  Int_t fCnew;          // flags for dynamical allocation
  Int_t fCdelete;       

  TString fTTBELL;

  TString fFlagSubDet;
  Int_t   fFlagPrint;

  TEcnaParEcal*   fEcal;

  //================================= BARREL

  Int_t** fT2dSMCrys;
  Int_t*  fT1dSMCrys;
  Int_t*  fT1dSMTow;
  Int_t*  fT1dTowEcha;

  TString fCodeChNumberingLvrbBot;
  TString fCodeChNumberingLvrbTop;

  //================================= ENDCAP

  Int_t*** fT3dDeeCrys;      // = CNA_CrysInDee[CNA_SCInDee - 1][CMS_CrysInSC - 1][CMS_DeeDir_index] 
  Int_t**  fT2dDeeCrys;
  Int_t*   fT1dDeeCrys;

  Int_t**  fT2dDeeSC;        // = CNA_SCInDee[CNA_CrysInDee - 1][CMS_DeeDir_index]
  Int_t*   fT1dDeeSC;

  Int_t**  fT2dSCEcha;       // = CMS_CrysInSC[CNA_CrysInDee - 1][CMS_DeeDir_index]
  Int_t*   fT1dSCEcha;

  Int_t**  fT2d_jch_JY;      // = JY[CMS_SCQuadType_index][CMS_CrysInSC - 1]
  Int_t*   fT1d_jch_JY;

  Int_t**  fT2d_ich_IX;      // = IX[CMS_SCQuadType_index][CMS_CrysInSC - 1]
  Int_t*   fT1d_ich_IX;

  TString  fCodeChNumberingITP1Bot;
  TString  fCodeChNumberingITP2Top;

  Int_t**  fT2d_DS;          // = DS[Dee - 1, CNA_SCInDee - 1]
  Int_t*   fT1d_DS;  

  Int_t**  fT2d_DSSC;        // = SCInDS[Dee - 1, CNA_SCInDee - 1]
  Int_t*   fT1d_DSSC;

  Int_t**  fT2d_DeeSCCons;   // = SCConsInDee[Dee - 1, CNA_SCInDee - 1]
  Int_t*   fT1d_DeeSCCons;

  Int_t**  fT2d_RecovDeeSC;  // = CNA_SCInDee[Dee - 1, SCConsInDee - 1]
  Int_t*   fT1d_RecovDeeSC;

  //........................................................................................

protected:

  void Init();

public:

  //------------------------------- methods

  TEcnaNumbering();
  TEcnaNumbering(TEcnaObject*, const TString&);
  TEcnaNumbering(const TString&, const TEcnaParEcal*);
  ~TEcnaNumbering();

  void SetEcalSubDetector(const TString&);
  void SetEcalSubDetector(const TString&, const TEcnaParEcal*);

  //================================== BARREL

  void  BuildBarrelCrysTable();   // correspondance crystal# in super-module <-> (tower number, channel# in tower)

  Int_t Get1SMCrysFrom1SMTowAnd0TowEcha(const Int_t&, const Int_t&);
  Int_t Get0SMEchaFrom1SMTowAnd0TowEcha(const Int_t&, const Int_t&);

  Int_t Get0TowEchaFrom0SMEcha(const Int_t&);
  Int_t Get1SMTowFrom0SMEcha(const Int_t&);
  Int_t Get0TowEchaFrom1SMCrys(const Int_t&);
  Int_t Get1SMTowFrom1SMCrys(const Int_t&);

  Int_t GetHashedNumberFromIEtaAndIPhi(const Int_t&, const Int_t&);
  Int_t GetIEtaFromHashed(const Int_t&, const Int_t&);
  Int_t GetIPhiFromHashed(const Int_t&);

  //..........................................................................................................
  Double_t GetEta(const Int_t&, const Int_t&, const Int_t&);
  Double_t GetEtaMin(const Int_t&, const Int_t&);
  Double_t GetEtaMax(const Int_t&, const Int_t&);

  Double_t GetIEtaMin(const Int_t&, const Int_t&);    // only for axis of TowerCrystalNumbering (+0.5 shift)
  Double_t GetIEtaMax(const Int_t&, const Int_t&);    // only for axis of TowerCrystalNumbering (-0.5 shift)

  Double_t GetIEtaMin(const Int_t&);                  // only for axis of SMTowerNumbering (+0.5 shift)
  Double_t GetIEtaMax(const Int_t&);                  // only for axis of SMTowerNumbering (-0.5 shift)

  Double_t GetPhiInSM(const Int_t&, const Int_t&, const Int_t&);
  Double_t GetPhi(const Int_t&, const Int_t&, const Int_t&);

  Double_t GetPhiMin(const Int_t&, const Int_t&);
  Double_t GetPhiMax(const Int_t&, const Int_t&);

  Double_t GetJPhiMin(const Int_t&, const Int_t&);
  Double_t GetJPhiMax(const Int_t&, const Int_t&);

  Double_t GetJPhiMin(const Int_t&);
  Double_t GetJPhiMax(const Int_t&);

  Double_t GetPhiMin(const Int_t&);
  Double_t GetPhiMax(const Int_t&);

  Double_t GetSMCentralPhi(const Int_t&);

  TString GetXDirectionEB(const Int_t&);
  TString GetYDirectionEB(const Int_t&);
  TString GetJYDirectionEB(const Int_t&);

  TString GetTowerLvrbType(const Int_t&);
  TString GetStinLvrbType(const Int_t&);

  TString GetSMHalfBarrel(const Int_t&);

  Int_t   PlusMinusSMNumber(const Int_t&);

  //================================== ENDCAP

  void BuildEndcapCrysTable(); // correspondance crystal# in Dee <-> (SC number, channel# in SC)
  void BuildEndcapSCTable();   // correspondance  SC# in Dee <-> (DS#, SC# in DS, SC# for construction)
                               // (SC = Super Crystal, DS = Data Sector)
 
  Int_t Get1DeeCrysFrom1DeeSCEcnaAnd0SCEcha(const Int_t&, const Int_t&, const TString&);

  Int_t Get1SCEchaFrom0DeeEcha(const Int_t&);
  Int_t Get1DeeSCEcnaFrom0DeeEcha(const Int_t&);
  Int_t Get1SCEchaFrom1DeeCrys(const Int_t&, const TString&);
  Int_t Get1DeeSCEcnaFrom1DeeCrys(const Int_t&, const TString&);

  Int_t GetDSFrom1DeeSCEcna(const Int_t&, const Int_t&);
  Int_t GetDSSCFrom1DeeSCEcna(const Int_t&, const Int_t&, const Int_t&);
  Int_t GetDSSCFrom1DeeSCEcna(const Int_t&, const Int_t&);
  Int_t GetDeeSCConsFrom1DeeSCEcna(const Int_t&, const Int_t&);
  Int_t GetDeeSCConsFrom1DeeSCEcna(const Int_t&, const Int_t&, const Int_t&);
  Int_t Get1DeeSCEcnaFromDeeSCCons(const Int_t&, const Int_t&);

  Int_t GetIXCrysInSC(const Int_t&, const Int_t&, const Int_t&);
  Int_t GetJYCrysInSC(const Int_t&, const Int_t&, const Int_t&);
  Int_t GetIXSCInDee(const Int_t&);
  Int_t GetJYSCInDee(const Int_t&);
  Int_t GetIXCrysInDee(const Int_t&, const Int_t&, const Int_t&);
  Int_t GetJYCrysInDee(const Int_t&, const Int_t&, const Int_t&);

  Int_t GetMaxSCInDS(const Int_t&);

  Double_t GetIIXMin(const Int_t&);    // only for axis of SCCrystalNumbering (+0.5 shift)
  Double_t GetIIXMax(const Int_t&);    // only for axis of SCCrystalNumbering (-0.5 shift)
  Double_t GetIIXMin();                // only for axis of DeeSCNumbering (+0.5 shift)
  Double_t GetIIXMax();                // only for axis of DeeSCNumbering (-0.5 shift)

  Double_t GetJIYMin(const Int_t&, const Int_t&);
  Double_t GetJIYMax(const Int_t&, const Int_t&);
  Double_t GetJIYMin(const Int_t&);
  Double_t GetJIYMax(const Int_t&);

  Int_t   GetDeeDirIndex(const TString&);
  Int_t   GetSCQuadTypeIndex(const TString&, const TString&);
  TString GetDeeDirViewedFromIP(const Int_t&);

  TString GetSCQuadFrom1DeeSCEcna(const Int_t&);
  TString GetEEDeeEndcap(const Int_t&);
  TString GetEEDeeType(const Int_t&);

  TString GetDeeHalfEndcap(const Int_t&);

  TString GetSCType(const Int_t&);  // for special (not connected) SC's

  Int_t StexEchaForCons(const Int_t&, const Int_t&);

  TString GetXDirectionEE(const Int_t&);
  TString GetYDirectionEE(const Int_t&);
  TString GetJYDirectionEE(const Int_t&);

  //========================================= BARREL and ENDCAP

  Int_t Get1StexStinFrom0StexEcha(const Int_t&);

  Int_t Get0StexEchaFrom1StexStinAnd0StinEcha(const Int_t&, const Int_t&);
  Int_t Get1StexCrysFrom1StexStinAnd0StinEcha(const Int_t&, const Int_t&, const Int_t&); // last arg = Stex Number

  Double_t GetIHocoMin(const Int_t&, const Int_t&);   // only for axis of TowerCrystalNumbering (+0.5 shift)
  Double_t GetIHocoMax(const Int_t&, const Int_t&);   // only for axis of TowerCrystalNumbering (-0.5 shift)

  Double_t GetVecoMin(const Int_t&, const Int_t&);
  Double_t GetVecoMax(const Int_t&, const Int_t&);

  Double_t GetJVecoMin(const Int_t&, const Int_t&);
  Double_t GetJVecoMax(const Int_t&, const Int_t&);

  TString GetStexHalfStas(const Int_t&);

  Int_t GetSMFromFED(const Int_t&);   // SM = SuperModule
  Int_t GetDSFromFED(const Int_t&);   // DS = DataSector

  Int_t MaxCrysInStinEcna(const Int_t&, const Int_t&, const TString&); // for not connected and incomplete SC's

  ClassDef(TEcnaNumbering,1)  //Channel Numbering for CNA
};
#endif
