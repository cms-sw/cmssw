// $Id: EcalPerEvtLaserAnalyzer.h

#include <memory>
#include <FWCore/Framework/interface/EDAnalyzer.h>

class TTree;
class TFile;

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
#define NSIDES      2     // Number of sides (0, 1) 

// "EB" geometry
#define NCRYSEB    1700  // Number of crystals per EB supermodule
#define NTTEB      68    // Number of EB Trigger Towers   
#define NPNEB      10    // Number of PN per EB supermodule

// "EE" geometry 
#define NCRYSEE    825   // Number of crystals per EE supermodule
#define NTTEE      33    // Number of EE Trigger Towers   
#define NPNEE      4     // Number of PN per EE supermodule



class EcalPerEvtLaserAnalyzer: public edm::EDAnalyzer{  

 public:
  
  explicit EcalPerEvtLaserAnalyzer(const edm::ParameterSet& iConfig);  
  ~EcalPerEvtLaserAnalyzer();
  
  
  virtual void analyze( const edm::Event & e, const  edm::EventSetup& c);
  virtual void beginJob();
  virtual void endJob();
    
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
  unsigned int  _niter ;
  int            _fedid ;
  unsigned int  _tower ;
  unsigned int  _channel ;
  std::string  _ecalPart;

  std::string  resdir_;
  std::string  refalphabeta_;
  std::string  digiCollection_;
  std::string  digiPNCollection_;
  std::string  digiProducer_;
  std::string  eventHeaderCollection_;
  std::string  eventHeaderProducer_;
      

  // Output file names

  std::string  alphafile;
  std::string  ADCfile;
  std::string  APDfile;
  std::string  resfile;


  //  Define geometrical constants
  //  Default values correspond to "EB" geometry (1700 crystals)

  unsigned int nCrys;
  unsigned int nTT;
  unsigned int nSides;


  int IsFileCreated;
  
  // Define number of channels (68 or 1700) for alpha and beta calculation

  unsigned int nCh;

  // Identify run type

  int runType;
  int runNum;
  int  dccID;
  int  fedID;
  int  lightside;

  int channelNumber;


  std::vector<int> colors;

  // Temporary root files and trees
  TFile *matacqFile;
  TTree *matacqTree; 

  TFile *ADCFile; 
  TTree *ADCtrees;

  TFile *APDFile; 
  TTree *header[2];
  TTree *APDtrees;


  int doesRefFileExist;
  int IsThereDataADC[nColor];
  int IsHeaderFilled[nColor];

  double ttMat, peakMat, peak;
  int evtMat, colMat;

  // Declaration of leaves types for temporary trees
  
  int             phi, eta;
  int             event ;
  int             color ;
  double          adc[10];
  int             adcG[10];
  double          tt;
  double          ttrig;
  double          pn0,pn1;
  double          pn[50];
  double          apdAmpl;
  double          apdTime;
  double          pnAmpl;



 

};


