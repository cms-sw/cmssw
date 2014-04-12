// $Id: EcalLaserAnalyzer2.h

#include <memory>
#include <FWCore/Framework/interface/EDAnalyzer.h>

class TFile;
class TTree;
class TProfile;
class TPNCor;
class TPN;
class TAPD;
class TMom;
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
#define NMODEE     21     // Number of EE submodules

#define NSIDES     2     // Number of sides
#define NREFCHAN   2     // Ref number for APDB
#define NSAMPSHAPES 250


class EcalLaserAnalyzer2: public edm::EDAnalyzer{  

 public:
  
  explicit EcalLaserAnalyzer2(const edm::ParameterSet& iConfig);  
  ~EcalLaserAnalyzer2();
  
  
  virtual void analyze( const edm::Event & e, const  edm::EventSetup& c);
  virtual void beginJob();
  virtual void endJob();
  
  void setGeomEB(int etaG, int phiG, int module, int tower, int strip, int xtal, 
		 int apdRefTT, int channel, int lmr);
  void setGeomEE(int etaG, int phiG,int iX, int iY, int iZ, int module, int tower, 
		 int ch , int apdRefTT, int channel, int lmr);

  enum VarCol  { iBlue, iRed, nColor }; 
  
 private:
  
  int iEvent;


  // Framework parameters
  
  unsigned int  _nsamples;
  unsigned int  _presample;
  unsigned int  _firstsample;
  unsigned int  _lastsample;
  unsigned int  _samplemin;
  unsigned int  _samplemax;
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
  double        _noise;
  std::string   _ecalPart;
  bool          _saveshapes;
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
  
  std::string  resdir_;
  std::string  digiCollection_;
  std::string  elecfile_;
  std::string  pncorfile_;
  std::string  digiPNCollection_;
  std::string  digiProducer_;
  std::string  eventHeaderCollection_;
  std::string  eventHeaderProducer_;
      

  // Output file names

  std::string  shapefile;
  std::string  matfile;
  std::string  ADCfile;
  std::string  APDfile;
  std::string  resfile;

  //  Define geometrical constants
  //  Default values correspond to "EB" geometry (1700 crystals)

  unsigned int nCrys;
  unsigned int nPNPerMod;
  unsigned int nRefChan;
  unsigned int nRefTrees;
  unsigned int nMod;
  unsigned int nSides;

  unsigned int nSamplesShapes;

  bool IsMatacqOK;

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

  std::vector<int> colors;
  std::map<int, int> channelMapEE;
  std::vector<int> dccMEM;
  std::vector<int> modules;
  std::map <int, unsigned int> apdRefMap[2];

  // PN linearity corrections

  TPNCor *pnCorrector;

  // get the shapes for amplitude determination

  bool getShapes();


  // Temporary root files and trees

  TFile *ADCFile; 
  TTree *ADCtrees[NCRYSEB];

  TFile *APDFile; 
  TTree *APDtrees[NCRYSEB];
  TTree *RefAPDtrees[NREFCHAN][NMODEE];

  TFile *resFile; 
  TTree *restrees[nColor];
  TTree *respntrees[nColor];

  TFile *ShapeFile; 
  TProfile *PulseShape;


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
  TPN  *PNFirstAnal[NMODEB][NPNPERMOD][nColor];
  TPN  *PNAnal[NMODEB][NPNPERMOD][nColor];

  // declare TAPD stuff
  TAPD *APDFirstAnal[NCRYSEB][nColor];
  TAPD *APDAnal[NCRYSEB][nColor];

  int IsThereDataADC[NCRYSEB][nColor];

  // Declaration of shapes

  std::vector<  double > shapesVec;
  double shapes[NSAMPSHAPES];
  double shapeCorrection;
  bool isMatacqOK;
  bool isSPRFine;


  // Declaration of leaves types for results tree

  int         pnID, moduleID, flag;
  int         channelIteratorEE;

  double      APD[6], Time[6], PN[6], APDoPN[6], APDoPNA[6], APDoPNB[6], 
    APDoAPDA[6], APDoAPDB[6], PNoPN[6], PNoPNA[6], PNoPNB[6], ShapeCor;
  // [0]=mean, [1]=rms, [2]=L3, [3]=nevt, [4]=min, [5]=max 
    

  double adcMean[NCRYSEB][10];
  int adcC[NCRYSEB];
  
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

  bool          isGainOK;
  bool          isTimingOK;


};


