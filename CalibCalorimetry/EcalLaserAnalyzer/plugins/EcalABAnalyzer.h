// $Id: EcalABAnalyzer.h

#include <memory>

#include <vector>
#include <map>

#include <FWCore/Framework/interface/EDAnalyzer.h>

using namespace std;

class TMom;
class TShapeAnalysis;
class TAPDPulse;

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

class EcalABAnalyzer: public edm::EDAnalyzer{  

 public:
  
  explicit EcalABAnalyzer(const edm::ParameterSet& iConfig);  
  ~EcalABAnalyzer();
  
  
  virtual void analyze( const edm::Event & e, const  edm::EventSetup& c);
  virtual void beginJob();
  virtual void endJob();


  enum VarCol   { iBlue, iRed, nColor }; 
  
 private:
  
  int iEvent;


  // Framework parameters
  
  string        _type; // LASER, LED

  unsigned int  _nsamples; // 10
  unsigned int  _presample; // default samples number for pedestal calc
  unsigned int  _firstsample; // samples number before max for ampl calc
  unsigned int  _lastsample; // samples number after max for ampl calc
  
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
 
  string        _ecalPart;
  int           _fedid;
  int           _debug;

  bool          _fitab ;
  double        _noise;
  double        _chi2cut;

  double        _alpha;
  double        _beta;
  unsigned int  _nevtmax;
  

  TAPDPulse *APDPulse;
  
  TMom *Delta01;
  TMom *Delta12;
  
  bool doesABTreeExist;
  bool doesABInitTreeExist;
  
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
  string  alphainitfile;

  // LASER/LED: AB

  TShapeAnalysis * shapana[nColor];
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
  int moduleID;
  int fedID;
  int dccID;
  int side;
  int lightside;
  int iZ;
  int channelIteratorEE;

  // Count Events of defined type
  int typeEvents;


  // Temporary root files and trees

  std::vector<int> colors;
  std::map<unsigned int, unsigned int> channelMapEE;
  std::vector<int> modules;

  // Declaration of leaves types for temporary trees
  
  int             phi, eta;
  int             event ;
  int             color ;
  double          adc[NSAMPAPD];
  int             adcGain;
  int             adcG[NSAMPAPD];

  double *adcNoPed;

  // Declaration of leaves types for results tree


  // Quality Checks variables and flags
  
  int nEvtBadGain[NCRYSEB];
  int nEvtBadTiming[NCRYSEB];
  int nEvtBadSignal[NCRYSEB];
  int nEvtTot[NCRYSEB];


  bool          wasGainOK[NCRYSEB];
  bool          wasTimingOK[NCRYSEB];
  bool          wasSignalOK[NCRYSEB];
  bool          wasABCalcOK[NCRYSEB];

  bool          isGainOK;
  bool          isTimingOK;
  bool          isSignalOK;

};


