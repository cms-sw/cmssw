#ifndef DQMOffline_Alignment_DiMuonMassBiasClient_h
#define DQMOffline_Alignment_DiMuonMassBiasClient_h
// -*- C++ -*-
//
// Package:    DQMOffline/Alignment
// Class  :    DiMuonMassBiasClient
//
// DQM class to plot di-muon mass bias in different kinematics bins

// system includes
#include <string>

// user includes
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace diMuonMassBias {

  struct fitOutputs {
  public:
    fitOutputs(const Measurement1D& bias, const Measurement1D& width) : m_bias(bias), m_width(width) {}

    // getters
    const Measurement1D getBias() { return m_bias; }
    const Measurement1D getWidth() { return m_width; }
    const bool isInvalid() {
      return (m_bias.value() == 0.f && m_bias.error() == 0.f && m_width.value() == 0.f && m_width.error() == 0.f);
    }

  private:
    Measurement1D m_bias;
    Measurement1D m_width;
  };

  // helper functions to fill arrays from vectors
  inline void fillArrayF(float* x, const edm::ParameterSet& cfg, const char* name) {
    auto v = cfg.getParameter<std::vector<double>>(name);
    assert(v.size() == 3);
    std::copy(std::begin(v), std::end(v), x);
  }

  inline void fillArrayI(int* x, const edm::ParameterSet& cfg, const char* name) {
    auto v = cfg.getParameter<std::vector<int>>(name);
    assert(v.size() == 3);
    std::copy(std::begin(v), std::end(v), x);
  }

  static constexpr int minimumHits = 10;

}  // namespace diMuonMassBias

class DiMuonMassBiasClient : public DQMEDHarvester {
public:
  /// Constructor
  DiMuonMassBiasClient(const edm::ParameterSet& ps);

  /// Destructor
  ~DiMuonMassBiasClient() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  /// BeginJob
  void beginJob(void) override;

  /// BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup) override;

  /// EndJob
  void dqmEndJob(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_) override;

private:
  /// book MEs
  void bookMEs(DQMStore::IBooker& ibooker);
  void getMEsToHarvest(DQMStore::IGetter& igetter);
  diMuonMassBias::fitOutputs fitLineShape(TH1* hist, const bool& fitBackground = false) const;
  void fitAndFill(std::pair<std::string, MonitorElement*> toHarvest, DQMStore::IBooker& iBooker);

  // data members
  const std::string TopFolder_;
  const bool fitBackground_;
  const bool useRooCBShape_;
  const bool useRooCMSShape_;
  const bool debugMode_;

  float meanConfig_[3];  /* parmaeters for the fit: mean */
  float widthConfig_[3]; /* parameters for the fit: width */
  float sigmaConfig_[3]; /* parameters for the fit: sigma */

  // list of histograms to harvest
  std::vector<std::string> MEtoHarvest_;

  // the histograms to be filled
  std::map<std::string, MonitorElement*> meanProfiles_;
  std::map<std::string, MonitorElement*> widthProfiles_;

  // the histograms than need to be fit and displays
  std::map<std::string, MonitorElement*> harvestTargets_;
};
#endif
