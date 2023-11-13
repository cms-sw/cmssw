#ifndef CONDCORE_CTPPSPLUGINS_PPSTIMINGCALIBRATIONPAYLOADINSPECTORHELPER_H
#define CONDCORE_CTPPSPLUGINS_PPSTIMINGCALIBRATIONPAYLOADINSPECTORHELPER_H

// User includes
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondFormats/PPSObjects/interface/PPSTimingCalibration.h"

// system includes
#include <memory>
#include <sstream>

// ROOT includes
#include "TCanvas.h"
#include "TStyle.h"
#include "TH2F.h"
#include "TLatex.h"
#include "TGraph.h"

namespace PPSTimingCalibrationPI {
  enum parameter { parameter0 = 0, parameter1 = 1, parameter2 = 2, parameter3 = 3 };

  inline std::string getStringFromParamEnum(const parameter& parameter) {
    const std::map<int, std::string> parameters = {{parameter0, "parameter 0"},
                                                   {parameter1, "parameter 1"},
                                                   {parameter2, "parameter 2"},
                                                   {parameter3, "parameter 3"}};

    auto it = parameters.find(parameter);
    if (it != parameters.end()) {
      return it->second;
    } else {
      return "no param";
    }
  }

  const std::string ARM = "db (0,1)";
  const std::string STATION = "station (1,2)";
  const std::string PLANE = "plane (0-3)";
  const std::string CHANNEL = "channel (0-11)";
}  // namespace PPSTimingCalibrationPI

/************************************************
    History plots
*************************************************/
template <PPSTimingCalibrationPI::parameter param, class PayloadType>
class ParametersPerRun : public cond::payloadInspector::HistoryPlot<PayloadType, float> {
public:
  ParametersPerRun()
      : cond::payloadInspector::HistoryPlot<PayloadType, float>(
            "Parameter " + PPSTimingCalibrationPI::getStringFromParamEnum(param) + " vs. Runs",
            PPSTimingCalibrationPI::getStringFromParamEnum(param)) {
    cond::payloadInspector::PlotBase::addInputParam(PPSTimingCalibrationPI::ARM);
    cond::payloadInspector::PlotBase::addInputParam(PPSTimingCalibrationPI::STATION);
    cond::payloadInspector::PlotBase::addInputParam(PPSTimingCalibrationPI::PLANE);
    cond::payloadInspector::PlotBase::addInputParam(PPSTimingCalibrationPI::CHANNEL);
  }

  float getFromPayload(PayloadType& payload) override {
    auto paramValues = cond::payloadInspector::PlotBase::inputParamValues();
    auto db = paramValues.find(PPSTimingCalibrationPI::ARM)->second;
    auto station = paramValues.find(PPSTimingCalibrationPI::STATION)->second;
    auto plane = paramValues.find(PPSTimingCalibrationPI::PLANE)->second;
    auto channel = paramValues.find(PPSTimingCalibrationPI::CHANNEL)->second;

    return payload.parameters(std::stoi(db), std::stoi(station), std::stoi(plane), std::stoi(channel))[param];
  }
};

/************************************************
    Image plots
*************************************************/
template <PPSTimingCalibrationPI::parameter param, class PayloadType>
class ParametersPerChannel : public cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV> {
public:
  ParametersPerChannel()
      : cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV>(
            "PPSTimingCalibration parameters per channel") {
    cond::payloadInspector::PlotBase::addInputParam(PPSTimingCalibrationPI::ARM);
    cond::payloadInspector::PlotBase::addInputParam(PPSTimingCalibrationPI::STATION);
    cond::payloadInspector::PlotBase::addInputParam(PPSTimingCalibrationPI::PLANE);
  }

  bool fill() override {
    auto tag = cond::payloadInspector::PlotBase::getTag<0>();
    auto tagname = tag.name;
    auto iov = tag.iovs.back();
    auto m_payload = this->fetchPayload(std::get<1>(iov));

    auto paramValues = cond::payloadInspector::PlotBase::inputParamValues();
    auto db = paramValues.find(PPSTimingCalibrationPI::ARM)->second;
    auto station = paramValues.find(PPSTimingCalibrationPI::STATION)->second;
    auto plane = paramValues.find(PPSTimingCalibrationPI::PLANE)->second;

    if (m_payload != nullptr) {
      TCanvas canvas(
          "PPSTimingCalibration parameters per channel", "PPSTimingCalibration parameters per channel", 1400, 1000);
      canvas.cd(1);
      canvas.SetGrid();
      const Int_t n = 12;
      Double_t x[n];
      Double_t y[n];
      for (int i = 0; i < n; i++) {
        y[i] = m_payload->parameters(std::stoi(db), std::stoi(station), std::stoi(plane), i)[param];
        x[i] = i;
      }

      TGraph* graph = new TGraph(n, x, y);
      graph->SetTitle(("PPSTimingCalibration db = " + db + ", " + "station = " + station + ", " + "plane = " + plane +
                       ", " + PPSTimingCalibrationPI::getStringFromParamEnum(param) + " PER channel; channel; " +
                       PPSTimingCalibrationPI::getStringFromParamEnum(param))
                          .c_str());
      graph->SetMarkerColor(2);
      graph->SetMarkerSize(1.5);
      graph->SetMarkerStyle(21);
      graph->GetXaxis()->SetRangeUser(-.5, 11.5);
      graph->GetXaxis()->SetNdivisions(16);
      graph->GetYaxis()->SetNdivisions(32);
      graph->Draw("AP");

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    } else {
      return false;
    }
  }
};

#endif