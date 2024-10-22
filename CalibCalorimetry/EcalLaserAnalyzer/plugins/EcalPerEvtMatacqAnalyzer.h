// $Id: EcalPerEvtMatacqAnalyzer.h

#include <memory>
#include <FWCore/Framework/interface/one/EDAnalyzer.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>

class TTree;
class TFile;

#define N_samples 2560
#define N_channels 1

class EcalPerEvtMatacqAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit EcalPerEvtMatacqAnalyzer(const edm::ParameterSet& iConfig);
  ~EcalPerEvtMatacqAnalyzer() override = default;

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void beginJob() override;
  void endJob() override;

private:
  std::string outfile;

  int iEvent;

  // Identify run type

  int runType;
  int runNum;

  //Declaration of leaves types

  int event;
  int laser_color;
  double matacq[N_samples];
  unsigned int maxsamp;
  unsigned int nsamples;

  TFile* sampFile;
  TTree* tree;

  int IsFileCreated;
  int IsTreeCreated;
  TFile* outFile;
  int status;
  double peak, sigma, fit, ampl, trise, ttrig;
  TTree* mtqShape;

private:
  //
  // Framework parameters
  //
  //  unsigned int _nsamples;
  const double _presample;
  const unsigned int _nsamplesaftmax;
  const unsigned int _nsamplesbefmax;
  const unsigned int _noiseCut;
  const unsigned int _parabnbefmax;
  const unsigned int _parabnaftmax;
  const unsigned int _thres;
  const unsigned int _lowlev;
  const unsigned int _highlev;
  const unsigned int _nevlasers;

  const std::string resdir_;
  const std::string digiCollection_;
  const std::string digiProducer_;
  const std::string eventHeaderCollection_;
  const std::string eventHeaderProducer_;

  const edm::EDGetTokenT<EcalMatacqDigiCollection> pmatToken_;
  const edm::EDGetTokenT<EcalRawDataCollection> dccToken_;
};
