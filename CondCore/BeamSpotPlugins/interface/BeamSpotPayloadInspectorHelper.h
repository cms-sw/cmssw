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

  std::pair<unsigned int, unsigned int> unpack(cond::Time_t since) {
    auto kLowMask = 0XFFFFFFFF;
    auto run = (since >> 32);
    auto lumi = (since & kLowMask);
    return std::make_pair(run, lumi);
  }

  enum parameters {
    X,
    Y,
    Z,
    sigmaX,
    sigmaY,
    sigmaZ,
    dxdz,
    dydz,
    lastLumi,
    lastRun,
    lastFill,
    nTracks,
    nPVs,
    creationTime,
    END_OF_TYPES
  };

  /************************************************/
  std::string getStringFromParamEnum(const parameters& parameter) {
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
          return std::make_pair<double, double>(payload.GetX(), payload.GetXError());
        case Y:
          return std::make_pair<double, double>(payload.GetY(), payload.GetYError());
        case Z:
          return std::make_pair<double, double>(payload.GetZ(), payload.GetZError());
        case sigmaX:
          return std::make_pair<double, double>(payload.GetBeamWidthX(), payload.GetBeamWidthXError());
        case sigmaY:
          return std::make_pair<double, double>(payload.GetBeamWidthY(), payload.GetBeamWidthYError());
        case sigmaZ:
          return std::make_pair<double, double>(payload.GetSigmaZ(), payload.GetSigmaZError());
        case dxdz:
          return std::make_pair<double, double>(payload.Getdxdz(), payload.GetdxdzError());
        case dydz:
          return std::make_pair<double, double>(payload.Getdydz(), payload.GetdydzError());
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
          return std::make_pair<double, double>(payload.GetX(), payload.GetXError());
        case Y:
          return std::make_pair<double, double>(payload.GetY(), payload.GetYError());
        case Z:
          return std::make_pair<double, double>(payload.GetZ(), payload.GetZError());
        case sigmaX:
          return std::make_pair<double, double>(payload.GetBeamWidthX(), payload.GetBeamWidthXError());
        case sigmaY:
          return std::make_pair<double, double>(payload.GetBeamWidthY(), payload.GetBeamWidthYError());
        case sigmaZ:
          return std::make_pair<double, double>(payload.GetSigmaZ(), payload.GetSigmaZError());
        case dxdz:
          return std::make_pair<double, double>(payload.Getdxdz(), payload.GetdxdzError());
        case dydz:
          return std::make_pair<double, double>(payload.Getdydz(), payload.GetdydzError());
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
          return std::make_pair<double, double>(payload.GetX(), payload.GetXError());
        case Y:
          return std::make_pair<double, double>(payload.GetY(), payload.GetYError());
        case Z:
          return std::make_pair<double, double>(payload.GetZ(), payload.GetZError());
        case sigmaX:
          return std::make_pair<double, double>(payload.GetBeamWidthX(), payload.GetBeamWidthXError());
        case sigmaY:
          return std::make_pair<double, double>(payload.GetBeamWidthY(), payload.GetBeamWidthYError());
        case sigmaZ:
          return std::make_pair<double, double>(payload.GetSigmaZ(), payload.GetSigmaZError());
        case dxdz:
          return std::make_pair<double, double>(payload.Getdxdz(), payload.GetdxdzError());
        case dydz:
          return std::make_pair<double, double>(payload.Getdydz(), payload.GetdydzError());
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
      return std::make_tuple(payload.GetX(), payload.GetY());
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
      canvas.cd(1)->SetRightMargin(0.03);
      canvas.cd(1)->Modified();
      canvas.cd(1)->SetGrid();

      auto h2_BSParameters = std::unique_ptr<TH2F>(new TH2F("Parameters", "", 2, 0.0, 2.0, 8, 0, 8.));
      h2_BSParameters->SetStats(false);

      std::function<double(parameters, bool)> cutFunctor = [this](parameters my_param, bool isError) {
        double ret(-999.);
        if (!isError) {
          switch (my_param) {
            case X:
              return m_payload->GetX();
            case Y:
              return m_payload->GetY();
            case Z:
              return m_payload->GetZ();
            case sigmaX:
              return m_payload->GetBeamWidthX();
            case sigmaY:
              return m_payload->GetBeamWidthY();
            case sigmaZ:
              return m_payload->GetSigmaZ();
            case dxdz:
              return m_payload->Getdxdz();
            case dydz:
              return m_payload->Getdydz();
            case END_OF_TYPES:
              return ret;
            default:
              return ret;
          }
        } else {
          switch (my_param) {
            case X:
              return m_payload->GetXError();
            case Y:
              return m_payload->GetYError();
            case Z:
              return m_payload->GetZError();
            case sigmaX:
              return m_payload->GetBeamWidthXError();
            case sigmaY:
              return m_payload->GetBeamWidthYError();
            case sigmaZ:
              return m_payload->GetSigmaZError();
            case dxdz:
              return m_payload->GetdxdzError();
            case dydz:
              return m_payload->GetdydzError();
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
        canvas.cd(2)->SetLeftMargin(0.15);
        canvas.cd(2)->SetRightMargin(0.03);
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
