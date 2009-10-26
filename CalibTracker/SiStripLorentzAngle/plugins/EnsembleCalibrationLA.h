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
  
  void write_ensembles_text(const Book&) const;
  void write_ensembles_plots(const Book&) const;
  void write_samples_plots(const Book&) const;

  const std::vector<std::string> inputFiles;
  const std::string inFileLocation, Prefix;
  const unsigned maxEvents,samples, nbins;
  const double lowBin,highBin;
  std::vector<int> vMethods;
};
}
#endif
