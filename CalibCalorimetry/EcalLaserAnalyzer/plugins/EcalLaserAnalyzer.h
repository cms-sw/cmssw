// $Id: EcalLaserAnalyzer.h

#include <memory>

#include <vector>
#include <map>

#include <FWCore/Framework/interface/EDAnalyzer.h>

class TFile;
class TTree;
class TProfile;
class TPNCor;
class TPN;
class TAPD;
class TMom;
class TShapeAnalysis;
class TAPDPulse;
class TPNPulse;
class TMem;

// Define geometrical constants 
// NOT the same for "EB" and "EE"
//
//     "EB"       "EE"
//
//      0          0
//   1     2    1     2
//   3     4
//   5     6
//   7     8 
//
// 

// "EB" geometry
#define NCRYSEB    1700  // Number of crystals per EB supermodule
#define NMODEB     9     // Number of EB submodules
#define NPNPERMOD  2     // Number of PN per module

// "EE" geometry 
#define NCRYSEE    830   // Number of crystals per EE supermodule
#define NMODEE     22     // Number of EE submodules

#define NSIDES     2     // Number of sides
#define NREFCHAN   2     // Ref number for APDB

class EcalLaserAnalyzer: public edm::EDAnalyzer{  

 public:
  
  explicit EcalLaserAnalyzer(const edm::ParameterSet& iConfig);  
  ~EcalLaserAnalyzer();
  
  
  virtual void analyze( const edm::Event & e, const  edm::EventSetup& c);
  virtual void beginJob();
  virtual void endJob();
  
  void setGeomEB(int etaG, int phiG, int module, int tower, int strip, int xtal, 
		 int apdRefTT, int channel, int lmr);
  void setGeomEE(int etaG, int phiG,int iX, int iY, int iZ, int module, int tower, 
		 int ch , int apdRefTT, int channel, int lmr);

  enum VarCol   { iBlue, iRed, nColor }; 
  
 private:
  
  int iEvent;


  // Framework parameters
  
  unsigned int  _nsamples;
  unsigned int  _presample;
  unsigned int  _firstsample;
  unsigned int  _lastsample;
  unsigned int  _nsamplesPN;
  unsigned int  _presamplePN;
  unsigned int  _firstsamplePN;
  unsigned int  _lastsamplePN;
  unsigned int  _timingcutlow;
  unsigned int  _timingcuthigh;
  unsigned int  _timingquallow;
  unsigned int  _timingqualhigh;
  double        _ratiomincutlow;
  double        _ratiomincuthigh;
  double        _ratiomaxcutlow;
  double        _presamplecut;
  unsigned int  _niter ;
  bool          _fitab ;
  double        _alpha;
  double        _beta;
  unsigned int  _nevtmax;
  double        _noise;
  double        _chi2cut;
  std::string   _ecalPart;
  bool          _docorpn;
  int           _fedid;
  bool          _saveallevents;
  double        _qualpercent;
  int           _debug;

  TAPDPulse *APDPulse;
  TPNPulse *PNPulse;
  TMem *Mem;
  TMom *Delta01;
  TMom *Delta12;
  
  bool doesABTreeExist;
  
  std::string  resdir_;
  std::string  pncorfile_;
  std::string  digiCollection_;
  std::string  digiPNCollection_;
  std::string  digiProducer_;
  std::string  eventHeaderCollection_;
  std::string  eventHeaderProducer_;

  // Output file names

  std::string  alphafile;
  std::string  alphainitfile;
  std::string  ADCfile;
  std::string  APDfile;
  std::string  resfile;

  
  TShapeAnalysis * shapana;
  unsigned int nevtAB[NCRYSEB];

  //  Define geometrical constants

  unsigned int nCrys;
  unsigned int nPNPerMod;
  unsigned int nRefChan;
  unsigned int nRefTrees;
  unsigned int nMod;
  unsigned int nSides;

  // Identify run type

  int runType;
  int runNum;

  // Identify channel

  int towerID;
  int channelID;
  int fedID;
  int dccID;
  int side;
  int lightside;
  int iZ;

  // Count Laser Events
  int laserEvents;

  // PN linearity corrections

  TPNCor *pnCorrector;

  // Temporary root files and trees

  TFile *ADCFile; 
  TTree *ADCtrees[NCRYSEB];

  TFile *APDFile; 
  TTree *APDtrees[NCRYSEB];
  TTree *RefAPDtrees[NREFCHAN][NMODEE];

  TFile *resFile; 
  TTree *restrees[nColor];
  TTree *respntrees[nColor];

  std::vector<int> colors;
  std::map<unsigned int, unsigned int> channelMapEE;
   std::vector<int> modules;
  std::map <int, unsigned int> apdRefMap[2];

  
  // Declaration of leaves types for temporary trees
  
  int             phi, eta;
  int             event ;
  int             color ;
  double          adc[10];
  int             adcG[10];
  double          pn0,pn1;
  double          pn[50];
  int             pnG[50];
  double          apdAmpl;
  double          apdAmplA;
  double          apdAmplB;
  double          apdTime;
  double          pnAmpl;

  int             eventref;
  int             colorref;

  double *adcNoPed;
  double *pnNoPed;

  // declare TPN stuff
  TPN  *PNFirstAnal[NMODEE][NPNPERMOD][nColor];
  TPN  *PNAnal[NMODEE][NPNPERMOD][nColor];

  // declare TAPD stuff
  TAPD *APDFirstAnal[NCRYSEB][nColor];
  TAPD *APDAnal[NCRYSEB][nColor];

  int IsThereDataADC[NCRYSEB][nColor];

  // Declaration of leaves types for results tree

  int         pnID, moduleID, flag, flagAB;
  int         channelIteratorEE;

  double      APD[6], Time[6], PN[6], APDoPN[6], APDoPNA[6], APDoPNB[6],
    APDoAPDA[6], APDoAPDB[6], PNoPN[6], PNoPNA[6], PNoPNB[6]; 

  // [0]=mean, [1]=rms, [2]=L3, [3]=nevt, [4]=min, [5]=max 
  // flag is 1 if fit if there is data, 0 if there is no data

  int iEta[NCRYSEB],iPhi[NCRYSEB];
  unsigned int iModule[NCRYSEB];
  int iTowerID[NCRYSEB],iChannelID[NCRYSEB], idccID[NCRYSEB], iside[NCRYSEB];
  unsigned int firstChanMod[NMODEE];
  unsigned int isFirstChanModFilled[NMODEE];

  // Quality Checks variables and flags
  
  int nEvtBadGain[NCRYSEB];
  int nEvtBadTiming[NCRYSEB];
  int nEvtTot[NCRYSEB];

  bool          wasGainOK[NCRYSEB];
  bool          wasTimingOK[NCRYSEB];
  bool          wasABCalcOK[NCRYSEB];

  bool          isGainOK;
  bool          isTimingOK;

};


