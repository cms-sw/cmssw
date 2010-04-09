// $Id: EcalCalibAnalyzer.h

#include <memory>

#include <vector>
#include <map>

#include <FWCore/Framework/interface/EDAnalyzer.h>

using namespace std;

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

#define NGAINPN    2     // Number of gains for TP   
#define NGAINAPD   4     // Number of gains for TP
#define NSAMPSHAPES 250

class EcalCalibAnalyzer: public edm::EDAnalyzer{  

 public:
  
  explicit EcalCalibAnalyzer(const edm::ParameterSet& iConfig);  
  ~EcalCalibAnalyzer();
  
  
  virtual void analyze( const edm::Event & e, const  edm::EventSetup& c);
  virtual void beginJob();
  virtual void endJob();

  void beginJobAB();
  void beginJobShape();
  void beginJobMatacq();

  void endJobAB();
  void endJobShape();
  void endJobLight();
  void endJobTestPulse();

  enum VarCol   { iBlue, iRed, nColor }; 
  
 private:
  
  int iEvent;


  // Framework parameters
  
  string        _type; // LASER, LED, TESTPULSE
  string        _typefit; // AB, SHAPE

  unsigned int  _nsamples; // 10
  unsigned int  _presample; // default samples number for pedestal calc
  unsigned int  _firstsample; // samples number before max for ampl calc
  unsigned int  _lastsample; // samples number after max for ampl calc
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
  double        _pulsemaxcutlow;
  double        _pulsemaxcuthigh;
  double        _qualpercent;
  double        _presamplecut;
  unsigned int  _niter ;
  double        _chi2cut;
  string        _ecalPart;
  int           _fedid;
  bool          _saveallevents;
  int           _debug;

  double        _noise;
  bool          _docorpn;

  bool          _fitab ;
  double        _alpha;
  double        _beta;
  unsigned int  _nevtmax;
  
  bool          _saveshapes;
  unsigned int  _nsamplesshapes;

  unsigned int  _samplemin;
  unsigned int  _samplemax;
  double        _chi2max ;
  double        _timeofmax ;

  TAPDPulse *APDPulse;
  TPNPulse *PNPulse;
  TMem *Mem;
  TMom *Delta01;
  TMom *Delta12;
  
  bool doesABTreeExist;
  bool doesABInitTreeExist;
  
  string  resdir_;
  string  pncorfile_;
  string  elecfile_;
  string  digiCollection_;
  string  digiPNCollection_;
  string  digiProducer_;
  string  eventHeaderCollection_;
  string  eventHeaderProducer_;

  // Output file names

  // LASER/LED: AB
  string  alphafile;
  string  alphainitfile;

  // LASER/LED: SHAPES
  string  shapefile;
  string  matfile;

  // COMMON:
  string  ADCfile;
  string  APDfile;
  string  resfile;

  
  // LASER/LED: AB

  TShapeAnalysis * shapana[nColor];
  unsigned int nevtAB[NCRYSEB];

  // LASER/LED: SHAPES

  vector<  double > shapesVec;
  double shapes[NSAMPSHAPES];
  double shapeCorrection;
  bool isSPRFine;
  bool getShapes();



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

  // Count Events of defined type
  int typeEvents;

  // PN linearity corrections

  TPNCor *pnCorrector;

  // Temporary root files and trees

  TFile *ADCFile; 
  TTree *ADCTrees[NCRYSEB];

  TFile *APDFile; 
  TTree *APDTrees[NCRYSEB];
  TTree *refAPDTrees[NREFCHAN][NMODEE];

  TFile *resFile; 
  TTree *resTrees[nColor];
  TTree *resPNTrees[nColor];
  TTree *resTree;
  TTree *resPNTree;

  TFile *shapeFile; 
  TProfile *pulseShape;

  std::vector<int> colors;
  std::map<unsigned int, unsigned int> channelMapEE;
   std::vector<int> modules;
  std::map <int, unsigned int> apdRefMap[2];

  
  // Declaration of leaves types for temporary trees
  
  int             phi, eta;
  int             event ;
  int             color ;
  double          adc[10];
  int             adcGain;
  int             adcG[10];
  double          pn0,pn1;
  double          pn[50];
  int             pnG[50];
  double          apdAmpl;
  double          apdAmplA;
  double          apdAmplB;
  double          apdTime;
  double          pnAmpl;
  int             pnGain;

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

  //declare stuff for TP:
  TPN  *TPPNAnal[NMODEE][NPNPERMOD][NGAINPN];
  TAPD  *TPAPDAnal[NCRYSEB][NGAINAPD];

  int isThereDataADC[NCRYSEB][nColor];

  // Declaration of leaves types for results tree

  int         pnID, moduleID, flag, flagAB;
  int         channelIteratorEE;

  bool isMatacqOK;

  double      APD[6], Time[6], PN[6], APDoPN[6], APDoPNA[6], APDoPNB[6],
    APDoAPD[6], APDoAPDA[6], APDoAPDB[6], PNoPN[6], PNoPNA[6], PNoPNB[6], ShapeCor; 

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
  int nEvtBadSignal[NCRYSEB];
  double meanRawAmpl[NCRYSEB];
  int nEvtRawAmpl[NCRYSEB];
  int nEvtTot[NCRYSEB];
  double meanMeanRawAmpl;

  unsigned int nGainPN; 
  unsigned int nGainAPD;

  bool          wasGainOK[NCRYSEB];
  bool          wasTimingOK[NCRYSEB];
  bool          wasSignalOK[NCRYSEB];
  bool          wasABCalcOK[NCRYSEB];

  bool          isGainOK;
  bool          isTimingOK;
  bool          isSignalOK;

};


