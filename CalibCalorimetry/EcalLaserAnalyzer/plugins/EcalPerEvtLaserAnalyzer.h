#ifndef CalibCalorimetry_EcalLaserAnalyzer_EcalPerEvtLaserAnalyzer_h
#define CalibCalorimetry_EcalLaserAnalyzer_EcalPerEvtLaserAnalyzer_h
// $Id: EcalPerEvtLaserAnalyzer.h

#include <memory>

#include <vector>

#include <FWCore/Framework/interface/one/EDAnalyzer.h>

#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <Geometry/EcalMapping/interface/EcalElectronicsMapping.h>
#include <Geometry/EcalMapping/interface/EcalMappingRcd.h>

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
#define NSIDES 2  // Number of sides (0, 1)

// "EB" geometry
#define NCRYSEB 1700  // Number of crystals per EB supermodule
#define NTTEB 68      // Number of EB Trigger Towers
#define NPNEB 10      // Number of PN per EB supermodule

// "EE" geometry
#define NCRYSEE 825  // Number of crystals per EE supermodule
#define NTTEE 33     // Number of EE Trigger Towers
#define NPNEE 4      // Number of PN per EE supermodule

class EcalPerEvtLaserAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit EcalPerEvtLaserAnalyzer(const edm::ParameterSet &iConfig);
  ~EcalPerEvtLaserAnalyzer() override;

  void analyze(const edm::Event &e, const edm::EventSetup &c) override;
  void beginJob() override;
  void endJob() override;

  enum VarCol { iBlue, iRed, nColor };

private:
  int iEvent;

  const std::string eventHeaderCollection_;
  const std::string eventHeaderProducer_;
  const std::string digiCollection_;
  const std::string digiProducer_;
  const std::string digiPNCollection_;

  const edm::EDGetTokenT<EcalRawDataCollection> rawDataToken_;
  edm::EDGetTokenT<EBDigiCollection> ebDigiToken_;
  edm::EDGetTokenT<EEDigiCollection> eeDigiToken_;
  const edm::EDGetTokenT<EcalPnDiodeDigiCollection> pnDiodeDigiToken_;
  const edm::ESGetToken<EcalElectronicsMapping, EcalMappingRcd> mappingToken_;

  // Framework parameters

  const unsigned int _nsamples;
  const unsigned int _presample;
  const unsigned int _firstsample;
  const unsigned int _lastsample;
  const unsigned int _nsamplesPN;
  const unsigned int _presamplePN;
  const unsigned int _firstsamplePN;
  const unsigned int _lastsamplePN;
  const unsigned int _timingcutlow;
  const unsigned int _timingcuthigh;
  const unsigned int _niter;
  const int _fedid;
  const unsigned int _tower;
  const unsigned int _channel;
  const std::string _ecalPart;

  const std::string resdir_;
  const std::string refalphabeta_;

  // Output file names

  std::string ADCfile;
  std::string resfile;

  //  Define geometrical constants
  //  Default values correspond to "EB" geometry (1700 crystals)

  unsigned int nCrys;

  int IsFileCreated;

  // Identify run type

  int runType;
  int runNum;
  int dccID;
  int fedID;
  int lightside;

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

  int phi, eta;
  int event;
  int color;
  double adc[10];
  int adcG[10];
  double tt;
  double ttrig;
  double pn0, pn1;
  double pn[50];
  double apdAmpl;
  double apdTime;
  double pnAmpl;
};
#endif
