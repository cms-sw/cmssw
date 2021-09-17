#ifndef CalibTracker_SiStripLorentzAngle_EnsembleCalibrationLA_h
#define CalibTracker_SiStripLorentzAngle_EnsembleCalibrationLA_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "CalibTracker/SiStripLorentzAngle/interface/LA_Filler_Fitter.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

namespace sistrip {
  class EnsembleCalibrationLA : public edm::EDAnalyzer {
  public:
    explicit EnsembleCalibrationLA(const edm::ParameterSet&);
    void analyze(const edm::Event&, const edm::EventSetup&) override {}
    void endRun(const edm::Run&, const edm::EventSetup&) override;
    void endJob() override;

  private:
    void write_ensembles_text(const Book&);
    void write_ensembles_plots(const Book&) const;
    void write_samples_plots(const Book&) const;
    void write_calibrations() const;

    const std::vector<std::string> inputFiles;
    const std::string inFileLocation, Prefix;
    const unsigned maxEvents, samples, nbins;
    const double lowBin, highBin;
    std::vector<int> vMethods;

    struct MethodCalibrations {
      MethodCalibrations()
          : slopes(std::vector<float>(14, 0)), offsets(std::vector<float>(14, 10)), pulls(std::vector<float>(14, 0)) {}
      std::vector<float> slopes;
      std::vector<float> offsets;
      std::vector<float> pulls;
    };
    std::map<std::string, MethodCalibrations> calibrations;

    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
    const TrackerTopology* tTopo_;
  };
}  // namespace sistrip
#endif
