#ifndef CalibCalorimetry_EcalLaserAnalyzer_EcalTestPulseAnalyzer_h
#define CalibCalorimetry_EcalLaserAnalyzer_EcalTestPulseAnalyzer_h

#include <memory>

#include <vector>
#include <map>

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

// "EB" geometry
#define NCRYSEB 1700  // Number of crystals per EB supermodule
#define NMODEB 9      // Number of EB submodules
#define NPNPERMOD 2   // Number of PN per module

// "EE" geometry
#define NCRYSEE 830  // Number of crystals per EE supermodule
#define NMODEE 21    // Number of EE submodules

#define NGAINPN 2   // Number of gains
#define NGAINAPD 4  // Number of gains

class EcalTestPulseAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit EcalTestPulseAnalyzer(const edm::ParameterSet &iConfig);
  ~EcalTestPulseAnalyzer() override;

  void analyze(const edm::Event &e, const edm::EventSetup &c) override;
  void beginJob() override;
  void endJob() override;

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
  const unsigned int _samplemin;
  const unsigned int _samplemax;
  const unsigned int _nsamplesPN;
  const unsigned int _presamplePN;
  const unsigned int _firstsamplePN;
  const unsigned int _lastsamplePN;
  const unsigned int _niter;
  const double _chi2max;
  const double _timeofmax;
  const std::string _ecalPart;
  const int _fedid;

  const std::string resdir_;

  // Output file names

  std::string rootfile;
  std::string resfile;

  //  Define geometrical constants
  //  Default values correspond to "EB" geometry (1700 crystals)

  unsigned int nCrys;
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

  TFile *outFile;  // from 'analyze': Data
  TFile *resFile;  // from 'endJob': Results

  // Temporary data trees

  TTree *trees[NCRYSEB];

  // Declaration of leaves types

  int phi, eta;
  int event;
  double adc[10];
  double pn[50];

  int apdGain;
  int pnGain;
  int pnG;
  double apdAmpl;
  double apdTime;
  double pnAmpl0;
  double pnAmpl1;
  double pnAmpl;

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

  int iEta[NCRYSEB], iPhi[NCRYSEB];
  unsigned int iModule[NCRYSEB];
  int iTowerID[NCRYSEB], iChannelID[NCRYSEB], idccID[NCRYSEB], iside[NCRYSEB];

  unsigned int firstChanMod[NMODEB];
  unsigned int isFirstChanModFilled[NMODEB];
};
#endif
