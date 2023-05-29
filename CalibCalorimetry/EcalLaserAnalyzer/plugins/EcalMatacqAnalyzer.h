// $Id: EcalMatacqAnalyzer.h

#include <memory>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <FWCore/Framework/interface/one/EDAnalyzer.h>

class TFile;
class TTree;
class TMTQ;

#define N_samples 2560
#define N_channels 1
#define NSIDES 2  // Number of sides

class EcalMatacqAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit EcalMatacqAnalyzer(const edm::ParameterSet &iConfig);
  ~EcalMatacqAnalyzer() override = default;

  void analyze(const edm::Event &e, const edm::EventSetup &c) override;
  void beginJob() override;
  void endJob() override;

  enum VarCol { iBlue, iRed, nColor };
  enum VarSide { iSide0, iSide1, nSide };

private:
  int iEvent;

  //
  // Framework parameters
  //

  const double _presample;
  const unsigned int _nsamplesaftmax;
  const unsigned int _noiseCut;
  const unsigned int _parabnbefmax;
  const unsigned int _parabnaftmax;
  const unsigned int _thres;
  const unsigned int _lowlev;
  const unsigned int _highlev;
  const unsigned int _nevlasers;
  const unsigned int _timebefmax;
  const unsigned int _timeaftmax;
  const double _cutwindow;
  const unsigned int _nsamplesshape;
  const unsigned int _presampleshape;
  const unsigned int _slide;
  const int _fedid;
  const int _debug;

  const std::string resdir_;
  const std::string digiCollection_;
  const std::string digiProducer_;
  const std::string eventHeaderCollection_;
  const std::string eventHeaderProducer_;
  const edm::EDGetTokenT<EcalMatacqDigiCollection> pmatToken_;
  const edm::EDGetTokenT<EcalRawDataCollection> dccToken_;

  std::string outfile;
  std::string sampfile;

  // Identify run type

  unsigned int nSides;
  int lightside;
  int runType;
  int runNum;
  int dccID;
  int fedID;

  // Count Laser Events
  int laserEvents;
  bool isThereMatacq;

  //Declaration of leaves types

  int event;
  int color;
  double matacq[N_samples];
  int maxsamp;
  int nsamples;
  double tt;

  TFile *sampFile;
  TTree *tree;

  TMTQ *MTQ[nColor][nSide];
  TTree *meanTree[nColor];

  std::vector<int> colors;

  TFile *outFile;
  int status;
  double peak, sigma, fit, ampl, trise, fwhm, fw20, fw80, ped, pedsig, ttrig, sliding;
  TTree *mtqShape;
};
