/*!
  \file RunInfo_PayloadInspector
  \Payload Inspector Plugin for RunInfo
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2018/03/18 10:01:00 $
*/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// helper
#include "CondCore/RunInfoPlugins/interface/RunInfoPayloadInspectoHelper.h"

// system includes
#include <memory>
#include <sstream>
#include <iostream>

// include ROOT
#include "TProfile.h"
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"
#include "TPaletteAxis.h"

namespace {

  using namespace cond::payloadInspector;

  /************************************************
     RunInfo Payload Inspector of 1 IOV 
  *************************************************/
  class RunInfoTest : public Histogram1D<RunInfo, SINGLE_IOV> {
  public:
    RunInfoTest() : Histogram1D<RunInfo, SINGLE_IOV>("Test RunInfo", "Test RunInfo", 1, 0.0, 1.0) {}

    bool fill() override {
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<RunInfo> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          payload->printAllValues();
        }
      }
      return true;
    }
  };

  /************************************************
    Summary of RunInfo of 1 IOV 
  *************************************************/
  class RunInfoParameters : public PlotImage<RunInfo, SINGLE_IOV> {
  public:
    RunInfoParameters() : PlotImage<RunInfo, SINGLE_IOV>("Display of RunInfo parameters") {}

    bool fill() override {
      gStyle->SetPaintTextFormat("g");
      auto tag = PlotBase::getTag<0>();
      auto iovs = tag.iovs;
      auto iov = iovs.front();
      std::shared_ptr<RunInfo> payload = fetchPayload(std::get<1>(iov));

      TCanvas canvas("RunInfo Parameters Summary", "RunInfo Parameters summary", 1000, 1000);
      canvas.cd();

      gStyle->SetHistMinimumZero();

      canvas.SetTopMargin(0.13);
      canvas.SetBottomMargin(0.06);
      canvas.SetLeftMargin(0.3);
      canvas.SetRightMargin(0.02);
      canvas.Modified();
      canvas.SetGrid();

      auto h2_RunInfoParameters = std::make_unique<TH2F>("Parameters", "", 1, 0.0, 1.0, 11, 0, 11.);
      auto h2_RunInfoState = std::make_unique<TH2F>("State", "", 1, 0.0, 1.0, 11, 0, 11.);
      h2_RunInfoParameters->SetStats(false);
      h2_RunInfoState->SetStats(false);

      std::function<float(RunInfoPI::parameters)> cutFunctor = [&payload](RunInfoPI::parameters my_param) {
        float ret(-999.);
        switch (my_param) {
          case RunInfoPI::m_run:
            return float(payload->m_run);
          case RunInfoPI::m_start_time_ll:
            return float(payload->m_start_time_ll * 1.0e-6);
          case RunInfoPI::m_stop_time_ll:
            return float(payload->m_stop_time_ll * 1.0e-6);
          case RunInfoPI::m_start_current:
            return payload->m_start_current;
          case RunInfoPI::m_stop_current:
            return payload->m_stop_current;
          case RunInfoPI::m_avg_current:
            return payload->m_avg_current;
          case RunInfoPI::m_max_current:
            return payload->m_max_current;
          case RunInfoPI::m_min_current:
            return payload->m_min_current;
          case RunInfoPI::m_run_interval_seconds:
            return RunInfoPI::runDuration(payload);
          case RunInfoPI::m_BField:
            return RunInfoPI::theBField(payload->m_avg_current);  //fieldIntensity;
          case RunInfoPI::m_fedIN:
            return float((payload->m_fed_in).size());
          case RunInfoPI::END_OF_TYPES:
            return ret;
          default:
            return ret;
        }
      };

      h2_RunInfoParameters->GetXaxis()->SetBinLabel(1, "Value");
      h2_RunInfoState->GetXaxis()->SetBinLabel(1, "Value");

      RunInfoPI::state theState;

      unsigned int yBin = 11;
      for (int foo = RunInfoPI::m_run; foo != RunInfoPI::END_OF_TYPES; foo++) {
        RunInfoPI::parameters param = static_cast<RunInfoPI::parameters>(foo);
        std::string theLabel = RunInfoPI::getStringFromTypeEnum(param);
        h2_RunInfoState->GetYaxis()->SetBinLabel(yBin, theLabel.c_str());
        h2_RunInfoParameters->GetYaxis()->SetBinLabel(yBin, theLabel.c_str());
        h2_RunInfoParameters->SetBinContent(1, yBin, cutFunctor(param));
        // non-fake payload
        if ((payload->m_run) != -1) {
          if ((payload->m_avg_current) <= -1) {
            // go in error state
            h2_RunInfoState->SetBinContent(1, yBin, 0.);
            theState = RunInfoPI::k_invalid;
          } else {
            // all is OK
            h2_RunInfoState->SetBinContent(1, yBin, 1.);
            theState = RunInfoPI::k_valid;
          }
        } else {
          // this is a fake payload
          h2_RunInfoState->SetBinContent(1, yBin, 0.9);
          theState = RunInfoPI::k_fake;
        }
        yBin--;
      }

      h2_RunInfoParameters->GetXaxis()->LabelsOption("h");
      h2_RunInfoParameters->GetYaxis()->SetLabelSize(0.05);
      h2_RunInfoParameters->GetXaxis()->SetLabelSize(0.05);
      h2_RunInfoParameters->SetMarkerSize(1.5);

      h2_RunInfoState->GetXaxis()->LabelsOption("h");
      h2_RunInfoState->GetYaxis()->SetLabelSize(0.05);
      h2_RunInfoState->GetXaxis()->SetLabelSize(0.05);
      h2_RunInfoState->SetMarkerSize(1.5);

      RunInfoPI::reportSummaryMapPalette(h2_RunInfoState.get());
      h2_RunInfoState->Draw("col");

      h2_RunInfoParameters->Draw("TEXTsame");

      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(12);
      t1.SetTextSize(0.03);
      t1.DrawLatex(0.1, 0.98, "RunInfo parameters:");
      t1.DrawLatex(0.1, 0.95, "payload:");
      t1.DrawLatex(0.1, 0.92, "start time:");
      t1.DrawLatex(0.1, 0.89, "end time:");

      t1.SetTextFont(42);
      t1.SetTextColor(4);
      t1.DrawLatex(0.37, 0.982, Form("IOV %s", std::to_string(+std::get<0>(iov)).c_str()));
      t1.DrawLatex(0.21, 0.952, Form(" %s", (std::get<1>(iov)).c_str()));
      t1.DrawLatex(0.23, 0.922, Form(" %s", (RunInfoPI::runStartTime(payload)).c_str()));
      t1.DrawLatex(0.23, 0.892, Form(" %s", (RunInfoPI::runEndTime(payload)).c_str()));

      TPaveText ksPt(0, 0, 0.35, 0.04, "NDC");
      ksPt.SetBorderSize(0);
      ksPt.SetFillColor(0);
      const char* textToAdd;
      switch (theState) {
        case RunInfoPI::k_fake:
          textToAdd = "This is a fake RunInfoPayload";
          break;
        case RunInfoPI::k_valid:
          textToAdd = "This is a valid RunInfoPayload";
          break;
        case RunInfoPI::k_invalid:
          textToAdd = "This is an invalid RunInfoPayload";
          break;
        default:
          throw cms::Exception("PayloadInspector") << "an invalid state has been found";
      }

      ksPt.AddText(textToAdd);
      ksPt.Draw();

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }
  };

  /************************************************
    time history of Magnet currents from RunInfo
  *************************************************/
  template <RunInfoPI::parameters param>
  class RunInfoCurrentHistory : public HistoryPlot<RunInfo, std::pair<bool, float> > {
  public:
    RunInfoCurrentHistory()
        : HistoryPlot<RunInfo, std::pair<bool, float> >(getStringFromTypeEnum(param),
                                                        getStringFromTypeEnum(param) + " value") {}
    ~RunInfoCurrentHistory() override = default;

    std::pair<bool, float> getFromPayload(RunInfo& payload) override {
      bool isRealRun = ((payload.m_run) != -1);

      switch (param) {
        case RunInfoPI::m_start_current:
          return std::make_pair(isRealRun, payload.m_start_current);
        case RunInfoPI::m_stop_current:
          return std::make_pair(isRealRun, payload.m_stop_current);
        case RunInfoPI::m_avg_current:
          return std::make_pair(isRealRun, payload.m_avg_current);
        case RunInfoPI::m_max_current:
          return std::make_pair(isRealRun, payload.m_max_current);
        case RunInfoPI::m_min_current:
          return std::make_pair(isRealRun, payload.m_min_current);
        case RunInfoPI::m_BField:
          return std::make_pair(isRealRun, RunInfoPI::theBField(payload.m_avg_current));
        default:
          edm::LogWarning("LogicError") << "Unknown parameter: " << param;
          break;
      }
      return std::make_pair(isRealRun, -1.0);
    }  // payload

    /************************************************/
    std::string getStringFromTypeEnum(const RunInfoPI::parameters& parameter) {
      switch (parameter) {
        case RunInfoPI::m_start_current:
          return "Magent start current [A]";
        case RunInfoPI::m_stop_current:
          return "Magnet stop current [A]";
        case RunInfoPI::m_avg_current:
          return "Magnet average current [A]";
        case RunInfoPI::m_max_current:
          return "Magnet max current [A]";
        case RunInfoPI::m_min_current:
          return "Magnet min current [A]";
        case RunInfoPI::m_BField:
          return "B-field intensity [T]";
        default:
          return "should never be here";
      }
    }
  };

  typedef RunInfoCurrentHistory<RunInfoPI::m_start_current> RunInfoStartCurrentHistory;
  typedef RunInfoCurrentHistory<RunInfoPI::m_stop_current> RunInfoStopCurrentHistory;
  typedef RunInfoCurrentHistory<RunInfoPI::m_avg_current> RunInfoAverageCurrentHistory;
  typedef RunInfoCurrentHistory<RunInfoPI::m_max_current> RunInfoMaxCurrentHistory;
  typedef RunInfoCurrentHistory<RunInfoPI::m_min_current> RunInfoMinCurrentHistory;
  typedef RunInfoCurrentHistory<RunInfoPI::m_BField> RunInfoBFieldHistory;

  /************************************************
    time history of Magnet currents from RunInfo
  *************************************************/
  template <RunInfoPI::DET theDet>
  class RunInfoDetHistory : public HistoryPlot<RunInfo, std::pair<bool, float> > {
  public:
    RunInfoDetHistory() : HistoryPlot<RunInfo, std::pair<bool, float> >("Det Status History", "Det Status History") {}
    ~RunInfoDetHistory() override = default;

    std::pair<bool, float> getFromPayload(RunInfo& payload) override {
      bool isRealRun = ((payload.m_run) != -1);
      float returnValue = 0.;
      // early return in case the run is fake
      if (!isRealRun) {
        return std::make_pair(isRealRun, returnValue);
      }

      const auto fedBounds = RunInfoPI::buildFEDBounds();
      const auto limits = fedBounds.at(theDet);
      const auto& FEDsIn = payload.m_fed_in;
      for (const auto& FED : FEDsIn) {
        if (FED > limits.first && FED < limits.second) {
          returnValue = 1.;
          break;  // break the loop to speed up time
        }
      }
      return std::make_pair(isRealRun, returnValue);
    }  // payload
  };

  using RunInfoTrackerHistory = RunInfoDetHistory<RunInfoPI::SISTRIP>;
  using RunInfoSiPixelHistory = RunInfoDetHistory<RunInfoPI::SIPIXEL>;
  using RunInfoSiPixelPhase1History = RunInfoDetHistory<RunInfoPI::SIPIXELPHASE1>;
  //.... other could be implemented
}  // namespace

PAYLOAD_INSPECTOR_MODULE(RunInfo) {
  PAYLOAD_INSPECTOR_CLASS(RunInfoTest);
  PAYLOAD_INSPECTOR_CLASS(RunInfoParameters);
  PAYLOAD_INSPECTOR_CLASS(RunInfoStopCurrentHistory);
  PAYLOAD_INSPECTOR_CLASS(RunInfoAverageCurrentHistory);
  PAYLOAD_INSPECTOR_CLASS(RunInfoMaxCurrentHistory);
  PAYLOAD_INSPECTOR_CLASS(RunInfoMinCurrentHistory);
  PAYLOAD_INSPECTOR_CLASS(RunInfoBFieldHistory);
  PAYLOAD_INSPECTOR_CLASS(RunInfoTrackerHistory);
  PAYLOAD_INSPECTOR_CLASS(RunInfoSiPixelHistory);
  PAYLOAD_INSPECTOR_CLASS(RunInfoSiPixelPhase1History);
}
