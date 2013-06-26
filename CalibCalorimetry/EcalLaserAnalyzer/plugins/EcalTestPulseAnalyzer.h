// $Id: EcalTestPulseAnalyzer.h,v 1.6 2012/02/09 10:07:37 eulisse Exp $

#include <memory>
#include <FWCore/Framework/interface/EDAnalyzer.h>

class TFile;
class TTree;

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
// "EB" geometry
#define NCRYSEB    1700  // Number of crystals per EB supermodule
#define NTTEB      68    // Number of EB Trigger Towers   
#define NMODEB     9     // Number of EB submodules
#define NPNPERMOD  2     // Number of PN per module

// "EE" geometry 
#define NCRYSEE    830   // Number of crystals per EE supermodule
#define NTTEE      68    // Number of EE Trigger Towers   
#define NMODEE     21     // Number of EE submodules

#define NGAINPN    2     // Number of gains   
#define NGAINAPD   4     // Number of gains 


class EcalTestPulseAnalyzer: public edm::EDAnalyzer{  

 public:
  
  explicit EcalTestPulseAnalyzer(const edm::ParameterSet& iConfig);  
  ~EcalTestPulseAnalyzer();
  
  
  virtual void analyze( const edm::Event & e, const  edm::EventSetup& c);
  virtual void beginJob();
  virtual void endJob();
  
  
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
  unsigned int  _niter ;
  double        _chi2max ;
  double        _timeofmax ;
  std::string   _ecalPart;
  int           _fedid;
  
  std::string  resdir_;
  std::string  digiCollection_;
  std::string  digiPNCollection_;
  std::string  digiProducer_;
  std::string  eventHeaderCollection_;
  std::string  eventHeaderProducer_;
   

  // Output file names

  std::string  rootfile;
  std::string  resfile;

  //  Define geometrical constants
  //  Default values correspond to "EB" geometry (1700 crystals)

  unsigned int nCrys;
  unsigned int nTT;
  unsigned int nMod;
  unsigned int nGainPN;
  unsigned int nGainAPD;

  
  // Count TP Events
  int TPEvents;

  double ret_data[20];

  int towerID;
  int channelID;

  // Identify run 

  int runType;
  int runNum;
  int fedID;
  int dccID;
  int side;
  int iZ;

    
  // Root Files
 
  TFile *outFile; // from 'analyze': Data
  TFile *resFile; // from 'endJob': Results


  // Temporary data trees 

  TTree *trees[NCRYSEB];

  // Declaration of leaves types
  
  int             phi, eta;
  int             event ;
  double          adc[10] ;
  double          pn[50] ;

  int             apdGain;
  int             pnGain;
  int             pnG;
  double          apdAmpl;
  double          apdTime;
  double          pnAmpl0;
  double          pnAmpl1;
  double          pnAmpl;


  // Results trees 

  TTree *restrees;
  TTree *respntrees;


  std::map<int, int> channelMapEE;
  std::vector<int> dccMEM;
  std::vector<int> modules;


  // Declaration of leaves types 

  int ieta, iphi, flag, gain;
  int pnID, moduleID;
  int channelIteratorEE;
  double APD[6], PN[6];

  int iEta[NCRYSEB],iPhi[NCRYSEB];
  unsigned int iModule[NCRYSEB];
  int iTowerID[NCRYSEB], iChannelID[NCRYSEB], idccID[NCRYSEB], iside[NCRYSEB];

  unsigned int firstChanMod[NMODEB];
  unsigned int isFirstChanModFilled[NMODEB];


};


