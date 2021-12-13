#ifndef CONDCORE_BEAMSPOTPLUGINS_BEAMSPOTPAYLOADINSPECTORHELPER_H
#define CONDCORE_BEAMSPOTPLUGINS_BEAMSPOTPAYLOADINSPECTORHELPER_H

// User includes

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"

// ROOT includes

#include <memory>
#include <sstream>
#include "TCanvas.h"
#include "TStyle.h"
#include "TH2F.h"
#include "TLatex.h"

namespace BeamSpotPI {

  inline std::pair<unsigned int, unsigned int> unpack(cond::Time_t since) {
    auto kLowMask = 0XFFFFFFFF;
    auto run = (since >> 32);
    auto lumi = (since & kLowMask);
    return std::make_pair(run, lumi);
  }

  enum parameters {
    X = 0,                // 0  regular BS methods
    Y = 1,                // 1
    Z = 2,                // 2
    sigmaX = 3,           // 3
    sigmaY = 4,           // 4
    sigmaZ = 5,           // 5
    dxdz = 6,             // 6
    dydz = 7,             // 7
    lastLumi = 8,         // 8  additional int's
    lastRun = 9,          // 9
    lastFill = 10,        // 10
    nTracks = 11,         // 11
    nPVs = 12,            // 12
    nUsedEvents = 13,     // 13
    maxPVs = 14,          // 14
    meanPV = 15,          // 15 additional float's
    meanErrorPV = 16,     // 16
    rmsPV = 17,           // 17
    rmsErrorPV = 18,      // 18
    creationTime = 19,    // 19 additional cond::Time_t
    startTimeStamp = 20,  // 20
    endTimeStamp = 21,    // 21
    startTime = 22,       // 22 additional std::string
    endTime = 23,         // 23
    lumiRange = 24,       // 24
    END_OF_TYPES = 25,
  };

  /************************************************/
  inline std::string getStringFromParamEnum(const parameters& parameter) {
    switch (parameter) {
      case X:
        return "X";
      case Y:
        return "Y";
      case Z:
        return "Z";
      case sigmaX:
        return "sigmaX";
      case sigmaY:
        return "sigmaY";
      case sigmaZ:
        return "sigmaZ";
      case dxdz:
        return "dx/dz";
      case dydz:
        return "dy/dz";
      default:
        return "should never be here";
    }
  }

  /************************************************
    template classes (history)
  *************************************************/

  template <parameters my_param, class PayloadType>
  class BeamSpot_history : public cond::payloadInspector::HistoryPlot<PayloadType, std::pair<double, double> > {
  public:
    BeamSpot_history()
        : cond::payloadInspector::HistoryPlot<PayloadType, std::pair<double, double> >(
              getStringFromParamEnum(my_param) + " vs run number", getStringFromParamEnum(my_param)) {}

    std::pair<double, double> getFromPayload(PayloadType& payload) override {
      auto ret = std::make_pair<double, double>(-9999., -9999.);

      switch (my_param) {
        case X:
          return std::make_pair<double, double>(payload.x(), payload.xError());
        case Y:
          return std::make_pair<double, double>(payload.y(), payload.yError());
        case Z:
          return std::make_pair<double, double>(payload.z(), payload.zError());
        case sigmaX:
          return std::make_pair<double, double>(payload.beamWidthX(), payload.beamWidthXError());
        case sigmaY:
          return std::make_pair<double, double>(payload.beamWidthY(), payload.beamWidthYError());
        case sigmaZ:
          return std::make_pair<double, double>(payload.sigmaZ(), payload.sigmaZError());
        case dxdz:
          return std::make_pair<double, double>(payload.dxdz(), payload.dxdzError());
        case dydz:
          return std::make_pair<double, double>(payload.dydz(), payload.dydzError());
        case END_OF_TYPES:
          return ret;
        default:
          return ret;
      }
    }
  };

  /************************************************
    template classes (run history)
   *************************************************/

  template <parameters my_param, class PayloadType>
  class BeamSpot_runhistory : public cond::payloadInspector::RunHistoryPlot<PayloadType, std::pair<double, double> > {
  public:
    BeamSpot_runhistory()
        : cond::payloadInspector::RunHistoryPlot<PayloadType, std::pair<double, double> >(
              getStringFromParamEnum(my_param) + " vs run number", getStringFromParamEnum(my_param)) {}

    std::pair<double, double> getFromPayload(PayloadType& payload) override {
      auto ret = std::make_pair<double, double>(-9999., -9999.);

      switch (my_param) {
        case X:
          return std::make_pair<double, double>(payload.x(), payload.xError());
        case Y:
          return std::make_pair<double, double>(payload.y(), payload.yError());
        case Z:
          return std::make_pair<double, double>(payload.z(), payload.zError());
        case sigmaX:
          return std::make_pair<double, double>(payload.beamWidthX(), payload.beamWidthXError());
        case sigmaY:
          return std::make_pair<double, double>(payload.beamWidthY(), payload.beamWidthYError());
        case sigmaZ:
          return std::make_pair<double, double>(payload.sigmaZ(), payload.sigmaZError());
        case dxdz:
          return std::make_pair<double, double>(payload.dxdz(), payload.dxdzError());
        case dydz:
          return std::make_pair<double, double>(payload.dydz(), payload.dydzError());
        case END_OF_TYPES:
          return ret;
        default:
          return ret;
      }
    }
  };

  /************************************************
    template classes (time history)
  *************************************************/

  template <parameters my_param, class PayloadType>
  class BeamSpot_timehistory : public cond::payloadInspector::TimeHistoryPlot<PayloadType, std::pair<double, double> > {
  public:
    BeamSpot_timehistory()
        : cond::payloadInspector::TimeHistoryPlot<PayloadType, std::pair<double, double> >(
              getStringFromParamEnum(my_param) + " vs time", getStringFromParamEnum(my_param)) {}

    std::pair<double, double> getFromPayload(PayloadType& payload) override {
      auto ret = std::make_pair<double, double>(-9999., -9999.);

      switch (my_param) {
        case X:
          return std::make_pair<double, double>(payload.x(), payload.xError());
        case Y:
          return std::make_pair<double, double>(payload.y(), payload.yError());
        case Z:
          return std::make_pair<double, double>(payload.z(), payload.zError());
        case sigmaX:
          return std::make_pair<double, double>(payload.beamWidthX(), payload.beamWidthXError());
        case sigmaY:
          return std::make_pair<double, double>(payload.beamWidthY(), payload.beamWidthYError());
        case sigmaZ:
          return std::make_pair<double, double>(payload.sigmaZ(), payload.sigmaZError());
        case dxdz:
          return std::make_pair<double, double>(payload.dxdz(), payload.dxdzError());
        case dydz:
          return std::make_pair<double, double>(payload.dydz(), payload.dydzError());
        case END_OF_TYPES:
          return ret;
        default:
          return ret;
      }
    }
  };

  /************************************************
    X-Y correlation plot
  *************************************************/
  template <class PayloadType>
  class xyCorrelation : public cond::payloadInspector::ScatterPlot<PayloadType, double, double> {
  public:
    xyCorrelation() : cond::payloadInspector::ScatterPlot<PayloadType, double, double>("BeamSpot x vs y", "x", "y") {}

    std::tuple<double, double> getFromPayload(PayloadType& payload) override {
      return std::make_tuple(payload.x(), payload.y());
    }
  };

  /************************************************
    Display of Beam Spot parameters
  *************************************************/
  template <class PayloadType>
  class DisplayParameters : public cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV> {
  public:
    DisplayParameters()
        : cond::payloadInspector::PlotImage<PayloadType, cond::payloadInspector::SINGLE_IOV>(
              "Display of BeamSpot parameters") {
      if constexpr (std::is_same_v<PayloadType, BeamSpotOnlineObjects>) {
        isOnline_ = true;
      } else {
        isOnline_ = false;
      }
    }

    bool fill() override {
      auto tag = cond::payloadInspector::PlotBase::getTag<0>();
      auto tagname = tag.name;
      auto iov = tag.iovs.front();

      gStyle->SetHistMinimumZero(kTRUE);

      m_payload = this->fetchPayload(std::get<1>(iov));

      TCanvas canvas("Beam Spot Parameters Summary", "BeamSpot Parameters summary", isOnline_ ? 1500 : 1000, 1000);
      if (isOnline_) {
        canvas.Divide(2, 1);
      }
      canvas.cd(1);

      canvas.cd(1)->SetTopMargin(0.05);
      canvas.cd(1)->SetBottomMargin(0.06);
      canvas.cd(1)->SetLeftMargin(0.15);
      canvas.cd(1)->SetRightMargin(0.01);
      canvas.cd(1)->Modified();
      canvas.cd(1)->SetGrid();

      auto h2_BSParameters = std::make_unique<TH2F>("Parameters", "", 2, 0.0, 2.0, 8, 0, 8.);
      h2_BSParameters->SetStats(false);

      std::function<double(parameters, bool)> cutFunctor = [this](parameters my_param, bool isError) {
        double ret(-999.);
        if (!isError) {
          switch (my_param) {
            case X:
              return m_payload->x();
            case Y:
              return m_payload->y();
            case Z:
              return m_payload->z();
            case sigmaX:
              return m_payload->beamWidthX();
            case sigmaY:
              return m_payload->beamWidthY();
            case sigmaZ:
              return m_payload->sigmaZ();
            case dxdz:
              return m_payload->dxdz();
            case dydz:
              return m_payload->dydz();
            case END_OF_TYPES:
              return ret;
            default:
              return ret;
          }
        } else {
          switch (my_param) {
            case X:
              return m_payload->xError();
            case Y:
              return m_payload->yError();
            case Z:
              return m_payload->zError();
            case sigmaX:
              return m_payload->beamWidthXError();
            case sigmaY:
              return m_payload->beamWidthYError();
            case sigmaZ:
              return m_payload->sigmaZError();
            case dxdz:
              return m_payload->dxdzError();
            case dydz:
              return m_payload->dydzError();
            case END_OF_TYPES:
              return ret;
            default:
              return ret;
          }
        }
      };

      h2_BSParameters->GetXaxis()->SetBinLabel(1, "Value");
      h2_BSParameters->GetXaxis()->SetBinLabel(2, "Error");

      unsigned int yBin = 8;
      for (int foo = parameters::X; foo <= parameters::dydz; foo++) {
        parameters param = static_cast<parameters>(foo);
        std::string theLabel = getStringFromTypeEnum(param);
        h2_BSParameters->GetYaxis()->SetBinLabel(yBin, theLabel.c_str());
        h2_BSParameters->SetBinContent(1, yBin, cutFunctor(param, false));
        h2_BSParameters->SetBinContent(2, yBin, cutFunctor(param, true));
        yBin--;
      }

      h2_BSParameters->GetXaxis()->LabelsOption("h");
      h2_BSParameters->GetYaxis()->SetLabelSize(0.05);
      h2_BSParameters->GetXaxis()->SetLabelSize(0.05);
      h2_BSParameters->SetMarkerSize(1.5);
      h2_BSParameters->Draw("TEXT");

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      if (isOnline_) {
        ltx.SetTextSize(0.030);
      } else {
        ltx.SetTextSize(0.025);
      }
      ltx.SetTextAlign(11);

      auto runLS = BeamSpotPI::unpack(std::get<0>(iov));

      ltx.DrawLatexNDC(
          gPad->GetLeftMargin(),
          1 - gPad->GetTopMargin() + 0.01,
          (tagname + " IOV: #color[4]{" + std::to_string(runLS.first) + "," + std::to_string(runLS.second) + "}")
              .c_str());

      if (isOnline_) {
        canvas.cd(2);
        canvas.cd(2)->SetTopMargin(0.05);
        canvas.cd(2)->SetBottomMargin(0.06);
        canvas.cd(2)->SetLeftMargin(0.25);
        canvas.cd(2)->SetRightMargin(0.01);
        canvas.cd(2)->Modified();
        canvas.cd(2)->SetGrid();

        auto extras = fillTheExtraHistogram();
        if (extras) {
          for (int bin = 1; bin <= extras->GetNbinsY(); bin++) {
            edm::LogVerbatim("BeamSpotPayloadInspectorHelper")
                << extras->GetYaxis()->GetBinLabel(bin) << ": " << extras->GetBinContent(1, bin) << "\n";
          }
        }
        extras->Draw("TEXT");

        ltx.DrawLatexNDC(
            gPad->GetLeftMargin(),
            1 - gPad->GetTopMargin() + 0.01,
            (tagname + " IOV: #color[4]{" + std::to_string(runLS.first) + "," + std::to_string(runLS.second) + "}")
                .c_str());

        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());

        return true;
      } else {
        std::string fileName(this->m_imageFileName);
        canvas.SaveAs(fileName.c_str());

        return true;
      }
    }

  public:
    virtual std::shared_ptr<TH2F> fillTheExtraHistogram() const { return nullptr; }

  protected:
    bool isOnline_;
    std::shared_ptr<PayloadType> m_payload;

    /************************************************/
    virtual std::string getStringFromTypeEnum(const parameters& parameter) const {
      switch (parameter) {
        case X:
          return "X [cm]";
        case Y:
          return "Y [cm]";
        case Z:
          return "Z [cm]";
        case sigmaX:
          return "#sigma_{X} [cm]";
        case sigmaY:
          return "#sigma_{Y} [cm]";
        case sigmaZ:
          return "#sigma_{Z} [cm]";
        case dxdz:
          return "#frac{dX}{dZ} [rad]";
        case dydz:
          return "#frac{dY}{dZ} [rad]";
        default:
          return "should never be here";
      }
    }
  };
}  // namespace BeamSpotPI

#endif
