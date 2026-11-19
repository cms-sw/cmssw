#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CondCore/HcalPlugins/interface/HcalObjRepresent.h"

// the data format of the condition to be inspected
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"

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

  class HcalPedestalContainer : public HcalObjRepresent::HcalDataContainer<HcalPedestals, HcalPedestal> {
  public:
    HcalPedestalContainer(std::shared_ptr<HcalPedestals> payload, unsigned int run)
        : HcalObjRepresent::HcalDataContainer<HcalPedestals, HcalPedestal>(payload, run) {}
    float getValue(const HcalPedestal* ped) override {
      return (ped->getValue(0) + ped->getValue(1) + ped->getValue(2) + ped->getValue(3)) / 4;
    }
  };

  /******************************************
     Detector map of HCAL Pedestals for 1 IOV
     (mode selects 2D eta/phi map, or eta/phi 1D profile)
  ******************************************/
  template <ViewMode Mode>
  class HcalPedestalsPlotT : public cond::payloadInspector::PlotImage<HcalPedestals> {
  public:
    HcalPedestalsPlotT() : cond::payloadInspector::PlotImage<HcalPedestals>("HCAL Pedestal - map") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto [time, hash] = iovs.front();
      std::shared_ptr<HcalPedestals> payload = fetchPayload(hash);
      if (!payload)
        return false;
      auto container = std::make_unique<HcalPedestalContainer>(payload, time);
      std::unique_ptr<TCanvas> canvas(container->getCanvasAll(modeName(Mode)));
      canvas->SaveAs(m_imageFileName.c_str());
      return true;
    }  // fill method
  };

  /**********************************************************
     Detector map of HCAL Pedestal difference between 2 IOVs
  **********************************************************/
  template <ViewMode Mode>
  class HcalPedestalsDiffT : public cond::payloadInspector::PlotImage<HcalPedestals> {
  public:
    HcalPedestalsDiffT() : cond::payloadInspector::PlotImage<HcalPedestals>("HCAL Pedestal difference") {
      setSingleIov(false);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash>>& iovs) override {
      auto [t1, h1] = iovs.front();
      auto [t2, h2] = iovs.back();

      std::shared_ptr<HcalPedestals> payload1 = fetchPayload(h1);
      std::shared_ptr<HcalPedestals> payload2 = fetchPayload(h2);
      if (!payload1 || !payload2)
        return false;

      auto container1 = std::make_unique<HcalPedestalContainer>(payload1, t1);
      auto container2 = std::make_unique<HcalPedestalContainer>(payload2, t2);
      container2->Subtract(container1.get());
      std::unique_ptr<TCanvas> canvas(container2->getCanvasAll(modeName(Mode)));
      canvas->SaveAs(m_imageFileName.c_str());
      return true;
    }  // fill method
  };

  using HcalPedestalsPlot = HcalPedestalsPlotT<ViewMode::Map>;
  using HcalPedestalsEtaPlot = HcalPedestalsPlotT<ViewMode::EtaProfile>;
  using HcalPedestalsPhiPlot = HcalPedestalsPlotT<ViewMode::PhiProfile>;
  using HcalPedestalsDiff = HcalPedestalsDiffT<ViewMode::Map>;
  using HcalPedestalsEtaDiff = HcalPedestalsDiffT<ViewMode::EtaProfile>;
  using HcalPedestalsPhiDiff = HcalPedestalsDiffT<ViewMode::PhiProfile>;

}  // namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(HcalPedestals) {
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalsPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalsDiff);
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalsPhiPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalsEtaPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalsEtaDiff);
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalsPhiDiff);
}
