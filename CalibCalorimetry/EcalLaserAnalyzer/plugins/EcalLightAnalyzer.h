// $Id: EcalLightAnalyzer.h

#include <memory>

#include <vector>
#include <map>

#include <FWCore/Framework/interface/EDAnalyzer.h>

using namespace std;

class TFile;
class TTree;
class TProfile;
class TH1D;
class TPN;
class TAPD;
class TMom;
class TShapeAnalysis;
class TAPDPulse;
class TPNPulse;
class TMem;
class TF1;
class TCalibData;

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
#define NPN        10     // Number of PN per module


// "EE" geometry 
#define NCRYSEE    830   // Number of crystals per EE supermodule
#define NMODEE     22     // Number of EE submodules
#define NMEMEE     2 

#define NSIDES     2     // Number of sides
#define NREFCHAN   2     // Ref number for APDB

#define NGAINPN    2     // Number of gains for TP   
#define NGAINAPD   4     // Number of gains for TP
#define NSAMPSHAPES 250

#define NSAMPAPD 10
#define NSAMPPN 50

#define VARSIZE 6  // Number of crystals per EB supermodule

class EcalLightAnalyzer: public edm::EDAnalyzer{  

 public:
  
  explicit EcalLightAnalyzer(const edm::ParameterSet& iConfig);  
  ~EcalLightAnalyzer();
  
  
  virtual void analyze( const edm::Event & e, const  edm::EventSetup& c);
  virtual void beginJob();
  virtual void endJob();



  enum VarCol   { iBlue, iRed, nColor }; 
  
 private:
  
  int iEvent;


  // Framework parameters
  
  string        _type; // LASER, LED
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
  double        _noise ;
  string        _ecalPart;
  int           _fedid;
  bool          _saveallevents;
  int           _debug;


  bool          _fitab ;
  double        _alpha;
  double        _beta;
  unsigned int  _nevtmax;
  
  unsigned int  _nsamplesshapes;
  unsigned int  _npresamplesshapes;
  double        _t0shapes;
  unsigned int  _samplemin;
  unsigned int  _samplemax;
  double        _chi2max ;
  double        _timeofmax ;

  TCalibData *calibData;
  TAPDPulse *APDPulse;
  TPNPulse *PNPulse;
  TMem *Mem;
  TMom *Delta01;
  TMom *Delta12;
  TH1D *laserShape[nColor][NSIDES];
  TH1D *pulseShape[nColor][NSIDES];
  
  
  string  resdir_;
  string  calibpath_;
  string  alphainitpath_;
  string  digiCollection_;
  string  digiPNCollection_;
  string  digiProducer_;
  string  eventHeaderCollection_;
  string  eventHeaderProducer_;

  // Output file names

  // LASER/LED: AB
  string  alphafile;

  // LASER/LED: SHAPES
  string  shapefile;
  string  matfile;

  // COMMON:
  string  ADCfile;
  string  APDfile;
  string  resfile;
  
  bool getConvolution(int ieta, int iphi, int icol);
  double getConvolutionPN(int iPN, int imem, int icol, int iside );
  bool getShapeCor(int ieta, int iphi, int icol, double &shapeCor_APD);

  bool getMatacq();
  bool getAB();

  // LASER/LED: SHAPES

  vector<  double > shapesAPD[nColor][NSIDES];
  void getLaserShapeForPN( int icol, int iside );
  double lasShapesForPN[nColor][NSIDES][NSAMPPN]; // FIXME: size for endcap
  bool gotLasShapeForPN[nColor][NSIDES];

  double _corrPNEB[NPN][nColor][NSIDES];
  double _corrPNEE[NPN][NMEMEE][nColor][NSIDES];


  double shapes[NSAMPSHAPES];
  double shapeCorrection;

  double sprmax( double t0, double t1, double t2, double tau1,
		 double tau2, int fed );
  double sprval( double t, double t0, double tau1, 
		 double tau2, double max, double fed );
  double sprvalled( double t, double t0, double tau1, 
		    double tau2, double max, double fed );
  TF1* spr( double t0, double t1, double t2, double tau1,
	    double tau2, int fed );


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

  std::vector<int> colors;
  std::vector<int> sides;
  std::map<unsigned int, unsigned int> channelMapEE;
   std::vector<int> modules;
  std::map <int, unsigned int> apdRefMap[2];

  
  // Declaration of leaves types for temporary trees
  
  int             phi, eta;
  int             event ;
  int             color ;
  double          adc[NSAMPAPD];
  int             adcGain;
  int             adcG[NSAMPAPD];
  double          pn0,pn1;
  double          pn0cor,pn1cor;
  double          pn[NSAMPPN];
  int             pnG[NSAMPPN];
  double          apdAmpl;
  double          apdAmplA;
  double          apdAmplB;
  double          apdTime;
  double          pnAmpl;
  double          pnAmplCor;
  int             pnGain;

  int             eventref;
  int             colorref;

  double *adcNoPed;
  double *pnNoPed;

  // declare TPN stuff
  TPN  *PNFirstAnal[NMODEE][NPNPERMOD][nColor];
  TPN  *PNAnal[NMODEE][NPNPERMOD][nColor];

  TPN  *PNFirstAnalCor[NMODEE][NPNPERMOD][nColor];
  TPN  *PNAnalCor[NMODEE][NPNPERMOD][nColor];

  // declare TAPD stuff
  TAPD *APDFirstAnal[NCRYSEB][nColor];
  TAPD *APDAnal[NCRYSEB][nColor];

  TAPD *APDFirstAnalCor[NCRYSEB][nColor];
  TAPD *APDAnalCor[NCRYSEB][nColor];

  int isThereDataADC[NCRYSEB][nColor];

  // Declaration of leaves types for results tree

  int         pnID, moduleID, flag, flagAB;
  int         channelIteratorEE;

  bool isMatacqOK;
  bool doesABTreeExist;

  double  APD[VARSIZE], Time[VARSIZE], PN[VARSIZE], 
    APDoPN[VARSIZE], APDoPNA[VARSIZE], APDoPNB[VARSIZE],
    APDoPNCor[VARSIZE], APDoPNACor[VARSIZE], APDoPNBCor[VARSIZE],
    APDoAPD[VARSIZE], APDoAPDA[VARSIZE], APDoAPDB[VARSIZE], 
    PNoPN[VARSIZE], PNoPNA[VARSIZE], PNoPNB[VARSIZE],
    shapeCorAPD,    shapeCorPN; 

  // [0]=mean, [1]=rms, [2]=L3, [3]=nevt, [4]=min, [5]=max 
  // flag is 1 if fit if there is data, 0 if there is no data

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

  bool          isGainOK;
  bool          isTimingOK;
  bool          isSignalOK;

};


