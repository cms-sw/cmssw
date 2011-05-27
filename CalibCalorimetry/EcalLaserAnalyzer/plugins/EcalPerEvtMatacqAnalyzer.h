// $Id: EcalPerEvtMatacqAnalyzer.h

#include <memory>
#include <FWCore/Framework/interface/EDAnalyzer.h>


class TTree;
class TFile;

#define N_samples 2560
#define N_channels 1



class EcalPerEvtMatacqAnalyzer: public edm::EDAnalyzer{  

 public:
  
  explicit EcalPerEvtMatacqAnalyzer(const edm::ParameterSet& iConfig);  
  ~EcalPerEvtMatacqAnalyzer();
  
  
  virtual void analyze( const edm::Event & e, const  edm::EventSetup& c);
  virtual void beginJob();
  virtual void endJob();
  
  
 private:
    
  std::string resdir_;
  std::string digiCollection_;
  std::string digiProducer_;
  std::string eventHeaderCollection_;
  std::string eventHeaderProducer_;

  std::string outfile;

  int iEvent;

  // Identify run type

  int runType;
  int runNum;

  //Declaration of leaves types
  
  int   event ;
  int laser_color ;
  double  matacq[N_samples]; 
  unsigned int  maxsamp; 
  unsigned int  nsamples; 

  TFile *sampFile;
  TTree *tree;

  int IsFileCreated;
  int IsTreeCreated;
  TFile *outFile;
  int status;
  double peak, sigma, fit, ampl, trise, ttrig;
  TTree* mtqShape;
  
 private:

//
// Framework parameters
//
//  unsigned int _nsamples;
  double       _presample;
  unsigned int _nsamplesaftmax;
  unsigned int _nsamplesbefmax;
  unsigned int _noiseCut;
  unsigned int _parabnbefmax;
  unsigned int _parabnaftmax;
  unsigned int _thres;
  unsigned int _lowlev;
  unsigned int _highlev;
  unsigned int _nevlasers;

};


