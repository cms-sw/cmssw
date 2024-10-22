#ifndef CalibTracker_SiStripESProducers_SiStripGainFactor_h
#define CalibTracker_SiStripESProducers_SiStripGainFactor_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"

class SiStripGainFactor {
public:
  SiStripGainFactor(const edm::ParameterSet& iConfig)
      : automaticMode_{iConfig.getParameter<bool>("AutomaticNormalization")},
        printdebug_{iConfig.getUntrackedParameter<bool>("printDebug", false)} {}

  void push_back_norm(double norm) { norm_.push_back(norm); }

  void resetIfBadNorm() {
    bool badNorm = std::find_if(norm_.begin(), norm_.end(), [](double x) { return x <= 0.; }) != norm_.end();

    if (!automaticMode_ && badNorm) {
      edm::LogError("SiStripGainESProducer") << "[SiStripGainESProducer] - ERROR: negative or zero Normalization "
                                                "factor provided. Assuming 1 for such factor"
                                             << std::endl;
      norm_ = std::vector<double>(norm_.size(), 1.);
    }
  }

  double get(const SiStripApvGain& gain, const int apvGainIndex) const {
    double NFactor = 0.;

    if (automaticMode_ || printdebug_) {
      std::vector<uint32_t> DetIds;
      gain.getDetIds(DetIds);

      double SumOfGains = 0.;
      int NGains = 0;
      for (uint32_t detid : DetIds) {
        SiStripApvGain::Range detRange = gain.getRange(detid);

        int iComp = 0;
        for (std::vector<float>::const_iterator apvit = detRange.first; apvit != detRange.second; apvit++) {
          SumOfGains += (*apvit);
          NGains++;
          if (printdebug_)
            edm::LogInfo("SiStripGainESProducer::produce()")
                << "detid/component: " << detid << "/" << iComp << "   gain factor " << *apvit;
          iComp++;
        }
      }

      if (automaticMode_) {
        if (SumOfGains > 0 && NGains > 0) {
          NFactor = SumOfGains / NGains;
        } else {
          edm::LogError(
              "SiStripGainESProducer::produce() - ERROR: empty set of gain values received. Cannot compute "
              "normalization factor. Assuming 1 for such factor")
              << std::endl;
          NFactor = 1.;
        }
      }
    }

    if (!automaticMode_) {
      NFactor = norm_[apvGainIndex];
    }

    if (printdebug_)
      edm::LogInfo("SiStripGainESProducer")
          << " putting A SiStrip Gain object in eventSetup with normalization factor " << NFactor;
    return NFactor;
  }

private:
  std::vector<double> norm_;
  bool automaticMode_;
  bool printdebug_;
};

#endif
