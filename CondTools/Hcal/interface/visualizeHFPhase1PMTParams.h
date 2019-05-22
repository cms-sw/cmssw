#ifndef CONDTOOLS_HCAL_VISUALIZEHFPHASE1PMTPARAMS_H_
#define CONDTOOLS_HCAL_VISUALIZEHFPHASE1PMTPARAMS_H_

#include <vector>
#include <sstream>

#include "CondTools/Hcal/interface/CmdLine.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/HcalObjects/interface/HFPhase1PMTParams.h"

struct VisualizationOptions {
  // Initialize the visualization options to some meaningful defaults
  double minCharge{-1.0e2};
  double maxCharge{1.0e3};
  double minTDC{0.0};
  double maxTDC{25.0};
  double minAsymm{-5.0};
  double maxAsymm{5.0};
  unsigned plotPoints{1000};
  bool verbose{false};

  void load(cmdline::CmdLine& cmdline) {
    cmdline.option(0, "--minAsymm") >> minAsymm;
    cmdline.option(0, "--maxAsymm") >> maxAsymm;
    cmdline.option(0, "--minCharge") >> minCharge;
    cmdline.option(0, "--maxCharge") >> maxCharge;
    cmdline.option(0, "--minTDC") >> minTDC;
    cmdline.option(0, "--maxTDC") >> maxTDC;
    cmdline.option(0, "--plotPoints") >> plotPoints;
    verbose = cmdline.has("-v");
  }

  static const char* options() {
    return "[-v] [--minAsymm a0] [--maxAsymm a1] [--minCharge Qmin] "
           "[--maxCharge Qmax] [--minTDC t0] [--maxTDC t1] [--plotPoints n]";
  }

  const char* description() const {
    if (descr_.empty()) {
      std::ostringstream os;

      os << "  --minAsymm   (default " << minAsymm << ") minimum value of charge asymmetry for plot axes\n\n";
      os << "  --maxAsymm   (default " << maxAsymm << ") maximum value of charge asymmetry for plot axes\n\n";
      os << "  --minCharge  (default " << minCharge << ") minimum value of charge for plot axes\n\n";
      os << "  --maxCharge  (default " << maxCharge << ") maximum value of charge for plot axes\n\n";
      os << "  --maxTDC     (default " << minTDC << ") minimum value of TDC time for plot axes\n\n";
      os << "  --maxTDC     (default " << maxTDC << ") maximum value of TDC time for plot axes\n\n";
      os << "  -v           (verbose) print various diagnostics "
         << "to the standard output\n";

      descr_ = os.str();
    }
    return descr_.c_str();
  }

private:
  mutable std::string descr_;
};

void visualizeHFPhase1PMTParams(const std::vector<HcalDetId>& idVec,
                                const HFPhase1PMTParams& cuts,
                                const VisualizationOptions& options,
                                const HFPhase1PMTParams* reference = nullptr);

#endif  // CONDTOOLS_HCAL_VISUALIZEHFPHASE1PMTPARAMS_H_
