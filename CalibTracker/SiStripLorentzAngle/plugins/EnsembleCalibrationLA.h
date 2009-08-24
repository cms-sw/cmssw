#ifndef CalibTracker_SiStripLorentzAngle_EnsembleCalibrationLA_h
#define CalibTracker_SiStripLorentzAngle_EnsembleCalibrationLA_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "CalibTracker/SiStripLorentzAngle/interface/LA_Filler_Fitter.h"

namespace sistrip {
class EnsembleCalibrationLA : public edm::EDAnalyzer {
 public:
  explicit EnsembleCalibrationLA(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) {}
  void endJob();
 private:
  std::vector<std::string> inputFiles;
  std::string inFileLocation, Prefix;
  unsigned maxEvents,samples, nbins;
  double lowBin,highBin;
  bool useWIDTH,useRATIO,useSQRTVAR;
};
}
#endif
