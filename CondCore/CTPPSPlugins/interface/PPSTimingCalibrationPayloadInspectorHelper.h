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
        return "parameter 0";
      case 1:
        return "parameter 1";
      case 2:
        return "parameter 2";
      case 3:
        return "parameter 3";

      default:
        return "not here";
    }
  }

  static std::string getStringFromDbEnum(const conditions_db& db) {
    switch (db) {
      case 0:
        return "db = 0";
      case 1:
        return "db = 1";

      default:
        return "not here";
    }
  }

  static std::string getStringFromPlaneEnum(const conditions_plane& plane) {
    switch (plane) {
      case 0:
        return "plane = 0";
      case 1:
        return "plane = 1";
      case 2:
        return "plane = 2";
      case 3:
        return "plane = 3";

      default:
        return "not here";
    }
  }

  static std::string getStringFromChannelEnum(const conditions_channel& channel) {
    switch (channel) {
      case 0:
        return "channel = 0";
      case 1:
        return "channel = 1";
      case 2:
        return "channel = 2";
      case 3:
        return "channel = 3";
      case 4:
        return "channel = 4";
      case 5:
        return "channel = 5";
      case 6:
        return "channel = 6";
      case 7:
        return "channel = 7";
      case 8:
        return "channel = 8";
      case 9:
        return "channel = 9";
      case 10:
        return "channel = 10";
      case 11:
        return "channel = 11";

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
            PPSTimingCalibrationPI::getStringFromParamEnum(param) + " " +
                PPSTimingCalibrationPI::getStringFromDbEnum(db) + " " +
                PPSTimingCalibrationPI::getStringFromPlaneEnum(plane) + " " +
                PPSTimingCalibrationPI::getStringFromChannelEnum(channel) + " vs. Runs",
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
            "TimingCalibration " + PPSTimingCalibrationPI::getStringFromParamEnum(param1) + " vs. " +
                PPSTimingCalibrationPI::getStringFromParamEnum(param2) + " on " +
                PPSTimingCalibrationPI::getStringFromDbEnum(db) + " " +
                PPSTimingCalibrationPI::getStringFromPlaneEnum(plane) + " " +
                PPSTimingCalibrationPI::getStringFromChannelEnum(channel),
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
    auto iov = tag.iovs.back();
    auto m_payload = this->fetchPayload(std::get<1>(iov));

    if (m_payload != nullptr) {
      TCanvas canvas(
          "PPSTimingCalibration parameters per channel", "PPSTimingCalibration parameters per channel", 1400, 1000);
      canvas.cd(1);
      canvas.SetGrid();
      const Int_t n = 12;
      Double_t x[n];
      Double_t y[n];
      for (int i = 0; i < n; i++) {
        y[i] = m_payload->parameters(db, 1, plane, i)[param];
        x[i] = i;
      }

      TGraph* graph = new TGraph(n, x, y);
      graph->SetTitle(("PPSTimingCalibration " + PPSTimingCalibrationPI::getStringFromDbEnum(db) + " " +
                       PPSTimingCalibrationPI::getStringFromPlaneEnum(plane) + " " +
                       PPSTimingCalibrationPI::getStringFromParamEnum(param) + " per channel; channel; parameter")
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
