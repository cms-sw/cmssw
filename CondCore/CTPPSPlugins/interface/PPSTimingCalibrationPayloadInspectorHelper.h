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

class PPSTimingCalibrationPI {
public:
  enum parameter { parameter0 = 0, parameter1 = 1, parameter2 = 2, parameter3 = 3 };

  enum conditions_db { db0 = 0, db1 = 1 };

  enum conditions_plane { plane0 = 0, plane1 = 1, plane2 = 2, plane3 = 3 };

  enum conditions_channel {
    channel0 = 0,
    channel1 = 1,
    channel2 = 2,
    channel3 = 3,
    channel4 = 4,
    channel5 = 5,
    channel6 = 6,
    channel7 = 7,
    channel8 = 8,
    channel9 = 9,
    channel10 = 10,
    channel11 = 11

  };

  static std::string getStringFromParamEnum(const parameter& parameter) {
    switch (parameter) {
      case 0:
        return "parameter no.0";
      case 1:
        return "parameter no.1";
      case 2:
        return "parameter no.2";
      case 3:
        return "parameter no.3";

      default:
        return "not here";
    }
  }
};

/************************************************
    History plots
  *************************************************/
template <PPSTimingCalibrationPI::conditions_db db,
          PPSTimingCalibrationPI::conditions_plane plane,
          PPSTimingCalibrationPI::conditions_channel channel,
          PPSTimingCalibrationPI::parameter param,
          class PayloadType>
class ParametersPerRun : public cond::payloadInspector::HistoryPlot<PayloadType, float> {
public:
  ParametersPerRun()
      : cond::payloadInspector::HistoryPlot<PayloadType, float>(
            PPSTimingCalibrationPI::getStringFromParamEnum(param) + "vs. Runs",
            PPSTimingCalibrationPI::getStringFromParamEnum(param)) {}

  float getFromPayload(PayloadType& payload) override { return payload.parameters(db, 1, plane, channel)[param]; }
};

/************************************************
    X-Y correlation plots
  *************************************************/
template <PPSTimingCalibrationPI::conditions_db db,
          PPSTimingCalibrationPI::conditions_plane plane,
          PPSTimingCalibrationPI::conditions_channel channel,
          PPSTimingCalibrationPI::parameter param1,
          PPSTimingCalibrationPI::parameter param2,
          class PayloadType>
class PpPCorrelation : public cond::payloadInspector::ScatterPlot<PayloadType, double, double> {
public:
  PpPCorrelation()
      : cond::payloadInspector::ScatterPlot<PayloadType, double, double>(
            "TimingCalibration" + PPSTimingCalibrationPI::getStringFromParamEnum(param1) + "vs." +
                PPSTimingCalibrationPI::getStringFromParamEnum(param2),
            PPSTimingCalibrationPI::getStringFromParamEnum(param1),
            PPSTimingCalibrationPI::getStringFromParamEnum(param2)) {}

  std::tuple<double, double> getFromPayload(PayloadType& payload) override {
    return std::make_tuple(payload.parameters(db, 1, plane, channel)[param1],
                           payload.parameters(db, 1, plane, channel)[param2]);
  }
};

/************************************************
    Other plots
  *************************************************/

template <PPSTimingCalibrationPI::conditions_db db,
          PPSTimingCalibrationPI::conditions_plane plane,
          PPSTimingCalibrationPI::parameter param,
          class PayloadType>
class ParametersPerChannel : public cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV> {
public:
  ParametersPerChannel()
      : cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV>(
            "PPSTimingCalibration parameters per channel") {}

  bool fill() override {
    auto tag = cond::payloadInspector::PlotBase::getTag<0>();
    auto tagname = tag.name;
    auto iov = tag.iovs.front();

    auto m_payload = this->fetchPayload(std::get<1>(iov));

    TCanvas canvas(
        "PPSTimingCalibration parameters per channel", "PPSTimingCalibration parameters per channel", 1400, 1000);
    canvas.cd(1);
    const Int_t n = 12;
    Double_t x[n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    Double_t y[n];
    for (int i = 0; i < n; i++) {
      y[i] = m_payload->parameters(db, 1, plane, i)[param];
    }

    TGraph* gr = new TGraph(n, x, y);
    gr->SetTitle("PPSTimingCalibration param per channel Example");
    gr->Draw("AP");

    std::string fileName(this->m_imageFileName);
    canvas.SaveAs(fileName.c_str());

    return true;
  }
};

#endif
