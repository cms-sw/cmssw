#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CondCore/HcalPlugins/interface/HcalObjRepresent.h"

// the data format of the condition to be inspected
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"

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

  class HcalPedestalWidthContainer : public HcalObjRepresent::HcalDataContainer<HcalPedestalWidths, HcalPedestalWidth> {
  public:
    HcalPedestalWidthContainer(std::shared_ptr<HcalPedestalWidths> payload, unsigned int run)
        : HcalObjRepresent::HcalDataContainer<HcalPedestalWidths, HcalPedestalWidth>(payload, run) {}
    float getValue(const HcalPedestalWidth* ped) override {
      return (ped->getWidth(0) + ped->getWidth(1) + ped->getWidth(2) + ped->getWidth(3)) / 4;
    }
  };

  /******************************************
     Detector map of HCAL PedestalWidths for 1 IOV
     (mode selects 2D eta/phi map, or eta/phi 1D profile)
  ******************************************/
  template <ViewMode Mode>
  class HcalPedestalWidthsPlotT : public cond::payloadInspector::PlotImage<HcalPedestalWidths> {
  public:
    HcalPedestalWidthsPlotT() : cond::payloadInspector::PlotImage<HcalPedestalWidths>("HCAL PedestalWidth - map") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto [time, hash] = iovs.front();
      std::shared_ptr<HcalPedestalWidths> payload = fetchPayload(hash);
      if (!payload)
        return false;
      auto container = std::make_unique<HcalPedestalWidthContainer>(payload, time);
      std::unique_ptr<TCanvas> canvas(container->getCanvasAll(modeName(Mode)));
      canvas->SaveAs(m_imageFileName.c_str());
      return true;
    }  // fill method
  };

  /**********************************************************
     Detector map of HCAL PedestalWidth difference between 2 IOVs
  **********************************************************/
  template <ViewMode Mode>
  class HcalPedestalWidthsDiffT : public cond::payloadInspector::PlotImage<HcalPedestalWidths> {
  public:
    HcalPedestalWidthsDiffT() : cond::payloadInspector::PlotImage<HcalPedestalWidths>("HCAL PedestalWidth difference") {
      setSingleIov(false);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto [t1, h1] = iovs.front();
      auto [t2, h2] = iovs.back();

      std::shared_ptr<HcalPedestalWidths> payload1 = fetchPayload(h1);
      std::shared_ptr<HcalPedestalWidths> payload2 = fetchPayload(h2);
      if (!payload1 || !payload2)
        return false;

      auto container1 = std::make_unique<HcalPedestalWidthContainer>(payload1, t1);
      auto container2 = std::make_unique<HcalPedestalWidthContainer>(payload2, t2);
      container2->Subtract(container1.get());
      std::unique_ptr<TCanvas> canvas(container2->getCanvasAll(modeName(Mode)));
      canvas->SaveAs(m_imageFileName.c_str());
      return true;
    }  // fill method
  };

  using HcalPedestalWidthsPlot = HcalPedestalWidthsPlotT<ViewMode::Map>;
  using HcalPedestalWidthsEtaPlot = HcalPedestalWidthsPlotT<ViewMode::EtaProfile>;
  using HcalPedestalWidthsPhiPlot = HcalPedestalWidthsPlotT<ViewMode::PhiProfile>;
  using HcalPedestalWidthsDiff = HcalPedestalWidthsDiffT<ViewMode::Map>;
  using HcalPedestalWidthsEtaDiff = HcalPedestalWidthsDiffT<ViewMode::EtaProfile>;
  using HcalPedestalWidthsPhiDiff = HcalPedestalWidthsDiffT<ViewMode::PhiProfile>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(HcalPedestalWidths) {
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalWidthsPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalWidthsDiff);
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalWidthsPhiPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalWidthsPhiDiff);
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalWidthsEtaPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalWidthsEtaDiff);
}
