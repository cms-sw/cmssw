#ifndef CalibCalorimetry_EcalLaserAnalyzer_EcalABAnalyzer_h
#define CalibCalorimetry_EcalLaserAnalyzer_EcalABAnalyzer_h

// $Id: EcalABAnalyzer.h

#include <memory>
#include <vector>
#include <map>

#include <FWCore/Framework/interface/one/EDAnalyzer.h>

#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <Geometry/EcalMapping/interface/EcalElectronicsMapping.h>
#include <Geometry/EcalMapping/interface/EcalMappingRcd.h>

class TShapeAnalysis;
class TAPDPulse;
class TMom;

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

// "EE" geometry
#define NCRYSEE 830  // Number of crystals per EE supermodule

class EcalABAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit EcalABAnalyzer(const edm::ParameterSet &iConfig);
  ~EcalABAnalyzer() override;

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

  const edm::EDGetTokenT<EcalRawDataCollection> rawDataToken_;
  edm::EDGetTokenT<EBDigiCollection> ebDigiToken_;
  edm::EDGetTokenT<EEDigiCollection> eeDigiToken_;
  const edm::ESGetToken<EcalElectronicsMapping, EcalMappingRcd> mappingToken_;

  // Framework parameters

  const unsigned int _nsamples;
  unsigned int _presample;
  const unsigned int _firstsample;
  const unsigned int _lastsample;
  const unsigned int _timingcutlow;
  const unsigned int _timingcuthigh;
  const unsigned int _timingquallow;
  const unsigned int _timingqualhigh;
  const double _ratiomincutlow;
  const double _ratiomincuthigh;
  const double _ratiomaxcutlow;
  const double _presamplecut;
  const unsigned int _niter;
  const double _alpha;
  const double _beta;
  const unsigned int _nevtmax;
  const double _noise;
  const double _chi2cut;
  const std::string _ecalPart;
  const int _fedid;
  const double _qualpercent;
  const int _debug;

  TAPDPulse *APDPulse;
  TMom *Delta01;
  TMom *Delta12;

  const std::string resdir_;

  // Output file names

  std::string alphafile;
  std::string alphainitfile;

  TShapeAnalysis *shapana;
  unsigned int nevtAB[NCRYSEB];

  //  Define geometrical constants
  //  Default values correspond to "EB" geometry (1700 crystals)

  unsigned int nCrys;
  bool doesABTreeExist;

  bool _fitab;
  // Identify run type

  int runType;
  int runNum;
  int fedID;
  int dccID;
  int side;
  int lightside;
  int iZ;

  // Temporary root files and trees

  std::vector<int> colors;
  std::map<int, int> channelMapEE;
  std::vector<int> dccMEM;
  std::vector<int> modules;

  // Declaration of leaves types for temporary trees

  int phi, eta;
  int event;
  int color;
  double adc[10];
  int adcG[10];
  int channelIteratorEE;

  int iEta[NCRYSEB], iPhi[NCRYSEB];
  int iTowerID[NCRYSEB], iChannelID[NCRYSEB], idccID[NCRYSEB], iside[NCRYSEB];

  // Quality Checks variables and flags

  int nEvtBadGain[NCRYSEB];
  int nEvtBadTiming[NCRYSEB];
  int nEvtTot[NCRYSEB];

  bool wasGainOK[NCRYSEB];
  bool wasTimingOK[NCRYSEB];

  bool isGainOK;
  bool isTimingOK;
};

#endif
