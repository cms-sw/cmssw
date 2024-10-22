#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTFormula.h"
#include "CondFormats/DataRecord/interface/PFCalibrationRcd.h"

#include <memory>
#include <sstream>
#include <fstream>
#include <iostream>
#include <array>
#include <map>

// include ROOT
#include "TH2F.h"
#include "TF1.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"

using namespace cond::payloadInspector;

class PerformancePayloadFromTFormulaExposed : public PerformancePayloadFromTFormula {
public:
  int resultPos(PerformanceResult::ResultType rt) const override {
    return PerformancePayloadFromTFormula::resultPos(rt);
  }
};

static std::map<PerformanceResult::ResultType, std::string> functType = {
    {PerformanceResult::PFfa_BARREL, "PFfa_BARREL"},
    {PerformanceResult::PFfa_ENDCAP, "PFfa_ENDCAP"},
    {PerformanceResult::PFfb_BARREL, "PFfb_BARREL"},
    {PerformanceResult::PFfb_ENDCAP, "PFfb_ENDCAP"},
    {PerformanceResult::PFfc_BARREL, "PFfc_BARREL"},
    {PerformanceResult::PFfc_ENDCAP, "PFfc_ENDCAP"},

    {PerformanceResult::PFfaEta_BARRELH, "PFfaEta_BARRELH"},
    {PerformanceResult::PFfaEta_ENDCAPH, "PFfaEta_ENDCAPH"},
    {PerformanceResult::PFfbEta_BARRELH, "PFfbEta_BARRELH"},
    {PerformanceResult::PFfbEta_ENDCAPH, "PFfbEta_ENDCAPH"},
    {PerformanceResult::PFfaEta_BARRELEH, "PFfaEta_BARRELEH"},
    {PerformanceResult::PFfaEta_ENDCAPEH, "PFfaEta_ENDCAPEH"},
    {PerformanceResult::PFfbEta_BARRELEH, "PFfbEta_BARRELEH"},
    {PerformanceResult::PFfbEta_ENDCAPEH, "PFfbEta_ENDCAPEH"},

    {PerformanceResult::PFfaEta_BARREL, "PFfaEta_BARREL"},
    {PerformanceResult::PFfaEta_ENDCAP, "PFfaEta_ENDCAP"},
    {PerformanceResult::PFfbEta_BARREL, "PFfbEta_BARREL"},
    {PerformanceResult::PFfbEta_ENDCAP, "PFfbEta_ENDCAP"},

    {PerformanceResult::PFfcEta_BARRELH, "PFfcEta_BARRELH"},
    {PerformanceResult::PFfcEta_ENDCAPH, "PFfcEta_ENDCAPH"},
    {PerformanceResult::PFfdEta_ENDCAPH, "PFfdEta_ENDCAPH"},
    {PerformanceResult::PFfcEta_BARRELEH, "PFfcEta_BARRELEH"},
    {PerformanceResult::PFfcEta_ENDCAPEH, "PFfcEta_ENDCAPEH"},
    {PerformanceResult::PFfdEta_ENDCAPEH, "PFfdEta_ENDCAPEH"}};

template <PerformanceResult::ResultType T>
class PfCalibration : public cond::payloadInspector::PlotImage<PerformancePayloadFromTFormula, SINGLE_IOV> {
public:
  PfCalibration()
      : cond::payloadInspector::PlotImage<PerformancePayloadFromTFormula, SINGLE_IOV>("Performance Payload formula") {}
  bool fill() override {
    auto tag = PlotBase::getTag<0>();
    auto iov = tag.iovs.front();
    std::string tagname = tag.name;
    auto payload = fetchPayload(std::get<1>(iov));

    if (!payload.get())
      return false;

    int pos = ((PerformancePayloadFromTFormulaExposed*)payload.get())->resultPos(T);
    auto formula = payload->formulaPayload();
    auto formula_vec = formula.formulas();
    auto limits_vec = formula.limits();
    if (pos < 0 || pos > (int)formula_vec.size()) {
      edm::LogError("PfCalibration") << "Will not display image for " << functType[T]
                                     << " as it's not contained in the payload!";
      return false;
    }
    TCanvas canvas("PfCalibration", "PfCalibration", 1500, 800);
    canvas.cd();
    auto formula_string = formula_vec[pos];
    auto limits = limits_vec[pos];

    auto function_plot = new TF1("f1", formula_string.c_str(), limits.first, limits.second);
    function_plot->SetTitle((functType[T] + " " + formula_string).c_str());
    function_plot->GetXaxis()->SetTitle("GeV");
    function_plot->Draw("");

    std::string fileName(m_imageFileName);
    canvas.SaveAs(fileName.c_str());

    return true;
  }
};

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(PerformancePayloadFromTFormula) {
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfa_BARREL>);
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfa_ENDCAP>);
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfb_BARREL>);
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfb_ENDCAP>);
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfc_BARREL>);
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfc_ENDCAP>);

  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfaEta_BARRELH>);
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfaEta_ENDCAPH>);
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfbEta_BARRELH>);
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfbEta_ENDCAPH>);
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfaEta_BARRELEH>);
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfaEta_ENDCAPEH>);
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfbEta_BARRELEH>);
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfbEta_ENDCAPEH>);

  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfaEta_BARREL>);
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfaEta_ENDCAP>);
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfbEta_BARREL>);
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfbEta_ENDCAP>);

  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfcEta_BARRELH>);
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfcEta_ENDCAPH>);
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfdEta_ENDCAPH>);
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfcEta_BARRELEH>);
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfcEta_ENDCAPEH>);
  PAYLOAD_INSPECTOR_CLASS(PfCalibration<PerformanceResult::PFfdEta_ENDCAPEH>);
}
