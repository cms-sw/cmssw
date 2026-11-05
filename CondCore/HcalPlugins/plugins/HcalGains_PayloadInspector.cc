#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CondCore/HcalPlugins/interface/HcalObjRepresent.h"

// the data format of the condition to be inspected
#include "CondFormats/HcalObjects/interface/HcalGains.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"
#include <memory>
#include <string>
#include <fstream>

namespace {

  using namespace HcalObjRepresent;

  class HcalGainContainer : public HcalObjRepresent::HcalDataContainer<HcalGains, HcalGain> {
  public:
    HcalGainContainer(std::shared_ptr<HcalGains> payload, unsigned int run)
        : HcalObjRepresent::HcalDataContainer<HcalGains, HcalGain>(payload, run) {}
    float getValue(const HcalGain* gain) override {
      return gain->getValue(0) + gain->getValue(1) + gain->getValue(2) + gain->getValue(3);
    }
  };

  /******************************************
     Detector map of HCAL Gains for 1 IOV
     (mode selects 2D eta/phi map, or eta/phi 1D profile)
  ******************************************/
  template <ViewMode Mode>
  class HcalGainsPlotT : public cond::payloadInspector::PlotImage<HcalGains> {
  public:
    HcalGainsPlotT() : cond::payloadInspector::PlotImage<HcalGains>("HCAL Gain - map") { setSingleIov(true); }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto [time, hash] = iovs.front();
      std::shared_ptr<HcalGains> payload = fetchPayload(hash);
      if (!payload)
        return false;
      auto container = std::make_unique<HcalGainContainer>(payload, time);
      std::unique_ptr<TCanvas> canvas(container->getCanvasAll(modeName(Mode)));
      canvas->SaveAs(m_imageFileName.c_str());
      return true;
    }  // fill method
  };

  /**********************************************************
     Detector map of HCAL Gain ratio between 2 IOVs
  **********************************************************/
  template <ViewMode Mode>
  class HcalGainsRatioT : public cond::payloadInspector::PlotImage<HcalGains> {
  public:
    HcalGainsRatioT() : cond::payloadInspector::PlotImage<HcalGains>("HCAL Gain Ratio") { setSingleIov(false); }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto [t1, h1] = iovs.front();
      auto [t2, h2] = iovs.back();

      std::shared_ptr<HcalGains> payload1 = fetchPayload(h1);
      std::shared_ptr<HcalGains> payload2 = fetchPayload(h2);
      if (!payload1 || !payload2)
        return false;

      auto container1 = std::make_unique<HcalGainContainer>(payload1, t1);
      auto container2 = std::make_unique<HcalGainContainer>(payload2, t2);
      container2->Divide(container1.get());
      std::unique_ptr<TCanvas> canvas(container2->getCanvasAll(modeName(Mode)));
      canvas->SaveAs(m_imageFileName.c_str());
      return true;
    }  // fill method
  };

  using HcalGainsPlot = HcalGainsPlotT<ViewMode::Map>;
  using HcalGainsEtaPlot = HcalGainsPlotT<ViewMode::EtaProfile>;
  using HcalGainsPhiPlot = HcalGainsPlotT<ViewMode::PhiProfile>;
  using HcalGainsRatio = HcalGainsRatioT<ViewMode::Map>;
  using HcalGainsEtaRatio = HcalGainsRatioT<ViewMode::EtaProfile>;
  using HcalGainsPhiRatio = HcalGainsRatioT<ViewMode::PhiProfile>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(HcalGains) {
  PAYLOAD_INSPECTOR_CLASS(HcalGainsPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalGainsRatio);
  PAYLOAD_INSPECTOR_CLASS(HcalGainsEtaPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalGainsPhiPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalGainsPhiRatio);
  PAYLOAD_INSPECTOR_CLASS(HcalGainsEtaRatio);
}
