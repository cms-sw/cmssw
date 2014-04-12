#include "TH1.h"
#include "TFile.h"
#include "TTree.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBBeamCounters.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBEventPosition.h"
#include <map>
#include <vector>

typedef struct TRIGGGER{
  int runNum;
  int eventNum;
  int beamTrigger;
  int fakeTrigger;
  int calibTrigger;
  int outSpillPedestalTrigger;
  int inSpillPedestalTrigger;
  int laserTrigger;
  int ledTrigger;
  int spillTrigger;
}TRIGGER;

typedef struct TDC{
  double trigger;
  double ttcL1;
  double beamCoincidence[5];
  double laserFlash;
  double qiePhase;
  double TOF1;
  double TOF2;
  double m1[5];
  double m2[5];
  double m3[5];
  double s1[5];
  double s2[5];
  double s3[5];
  double s4[5];
  double bh1[5];
  double bh2[5];
  double bh3[5];
  double bh4[5];
}TDC;

typedef struct ADC{
  double    VM;
  double    V3;
  double    V6;
  double    VH1;
  double    VH2;
  double    VH3;
  double    VH4;
  double    Ecal7x7;
  double    Sci521;
  double    Sci528;
  double    CK1;
  double    CK2;
  double    CK3;
  double    SciVLE;
  double    S1;
  double    S2;
  double    S3;
  double    S4;
  double    VMF;
  double    VMB;
  double    VM1;
  double    VM2;
  double    VM3;
  double    VM4;
  double    VM5;
  double    VM6;
  double    VM7;
  double    VM8;
  double    TOF1;
  double    TOF2;
  double    BH1;
  double    BH2;
  double    BH3;
  double    BH4;
}ADC;

typedef struct CHAMB{
  double WCAx[5];
  double WCAy[5];
  double WCBx[5];
  double WCBy[5];
  double WCCx[5];
  double WCCy[5];
  double WCDx[5];
  double WCDy[5];
  double WCEx[5];
  double WCEy[5];
  double WCFx[5];
  double WCFy[5];
  double WCGx[5];
  double WCGy[5];
  double WCHx[5];
  double WCHy[5];
}CHAMB; 


typedef struct ZDCN{
  double zdcHADMod1;
  double zdcHADMod2;
  double zdcHADMod3;
  double zdcHADMod4;
  double zdcEMMod1;
  double zdcEMMod2;
  double zdcEMMod3;
  double zdcEMMod4;
  double zdcEMMod5;
  double zdcScint1;
  double zdcScint2;
  double zdcExtras[7];
}ZDCN;

typedef struct ZDCP{
  double zdcHADMod1;
  double zdcHADMod2;
  double zdcHADMod3;
  double zdcHADMod4;
  double zdcEMMod1;
  double zdcEMMod2;
  double zdcEMMod3;
  double zdcEMMod4;
  double zdcEMMod5;
  double zdcScint1;
  double zdcScint2;
  double zdcExtras[7];
}ZDCP;

class ZdcTBAnalysis {
public:
  ZdcTBAnalysis(); 
  void setup(const std::string& histoFileName);
  void analyze(const ZDCRecHitCollection& hf);
  void analyze(const HcalTBTriggerData& trg);
  void analyze(const HcalTBBeamCounters& bc);
  void analyze(const HcalTBTiming& times);
  void analyze(const HcalTBEventPosition& chpos);
  void fillTree();
  void done();

 private:
  int iside;
  int isection;
  int ichannel;
  int idepth;
  double energy;
  HcalZDCDetId detID;

  int runNumber;
  int eventNumber;
  bool isBeamTrigger;
  bool isFakeTrigger;
  bool isCalibTrigger;
  bool isOutSpillPedestalTrigger;
  bool isInSpillPedestalTrigger;
  bool isLaserTrigger;
  bool isLedTrigger;
  bool isSpillTrigger;

  double trigger_time;
  double ttc_L1a_time;
  double beam_coincidence[5];
  double laser_flash;
  double qie_phase;
  double TOF1_time;
  double TOF2_time;

  double m1hits[5];
  double m2hits[5];
  double m3hits[5];
  double s1hits[5];
  double s2hits[5];
  double s3hits[5];
  double s4hits[5];
  double bh1hits[5];
  double bh2hits[5];
  double bh3hits[5];
  double bh4hits[5];
  
  double    VMadc;
  double    V3adc;
  double    V6adc;
  double    VH1adc;
  double    VH2adc;
  double    VH3adc;
  double    VH4adc;
  double    Ecal7x7adc;
  double    Sci521adc;
  double    Sci528adc;

  double    CK1adc;
  double    CK2adc;
  double    CK3adc;
  double    SciVLEadc;
  double    S1adc;
  double    S2adc;
  double    S3adc;
  double    S4adc;

  double    VMFadc;
  double    VMBadc;
  double    VM1adc;
  double    VM2adc;
  double    VM3adc;
  double    VM4adc;
  double    VM5adc;
  double    VM6adc;
  double    VM7adc;
  double    VM8adc;
  double    TOF1adc;
  double    TOF2adc;
  double    BH1adc;
  double    BH2adc;
  double    BH3adc;
  double    BH4adc;

  std::vector<double> wcax;
  std::vector<double> wcay;
  std::vector<double> wcbx;
  std::vector<double> wcby;
  std::vector<double> wccx;
  std::vector<double> wccy;
  std::vector<double> wcdx;
  std::vector<double> wcdy;
  std::vector<double> wcex;
  std::vector<double> wcey;
  std::vector<double> wcfx;
  std::vector<double> wcfy;
  std::vector<double> wcgx;
  std::vector<double> wcgy;
  std::vector<double> wchx;
  std::vector<double> wchy;

  TRIGGER trigger;
  TDC tdc;
  ADC adc;
  CHAMB chamb;
  ZDCP zdcp;
  ZDCN zdcn;

  TFile* outFile; 
  TTree* ZdcAnalize;

 };
