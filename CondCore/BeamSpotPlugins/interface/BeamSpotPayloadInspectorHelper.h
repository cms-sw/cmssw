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

  /************************************************
    Display of Beam Spot parameters difference
  *************************************************/
  template <class PayloadType, cond::payloadInspector::IOVMultiplicity nIOVs, int ntags>
  class DisplayParametersDiff : public cond::payloadInspector::PlotImage<PayloadType, nIOVs, ntags> {
  public:
    DisplayParametersDiff()
        : cond::payloadInspector::PlotImage<PayloadType, nIOVs, ntags>("Display of BeamSpot parameters differences") {
      if constexpr (std::is_same_v<PayloadType, BeamSpotOnlineObjects>) {
        isOnline_ = true;
      } else {
        isOnline_ = false;
      }
    }

    bool fill() override {
      // trick to deal with the multi-ioved tag and two tag case at the same time
      auto theIOVs = cond::payloadInspector::PlotBase::getTag<0>().iovs;
      auto f_tagname = cond::payloadInspector::PlotBase::getTag<0>().name;
      std::string l_tagname = "";
      auto firstiov = theIOVs.front();
      std::tuple<cond::Time_t, cond::Hash> lastiov;

      // we don't support (yet) comparison with more than 2 tags
      assert(this->m_plotAnnotations.ntags < 3);

      if (this->m_plotAnnotations.ntags == 2) {
        auto tag2iovs = cond::payloadInspector::PlotBase::getTag<1>().iovs;
        l_tagname = cond::payloadInspector::PlotBase::getTag<1>().name;
        lastiov = tag2iovs.front();
      } else {
        lastiov = theIOVs.back();
      }

      l_payload = this->fetchPayload(std::get<1>(lastiov));
      f_payload = this->fetchPayload(std::get<1>(firstiov));

      std::string lastIOVsince = std::to_string(std::get<0>(lastiov));
      std::string firstIOVsince = std::to_string(std::get<0>(firstiov));

      TCanvas canvas("Beam Spot Parameters Difference Summary", "Beam Spot Parameters Difference summary", 1000, 1000);
      canvas.cd(1);

      canvas.cd(1)->SetTopMargin(0.08);
      canvas.cd(1)->SetBottomMargin(0.06);
      canvas.cd(1)->SetLeftMargin(0.14);
      canvas.cd(1)->SetRightMargin(0.16);
      canvas.cd(1)->Modified();
      canvas.cd(1)->SetGrid();

      auto h2_BSParameters = std::make_unique<TH2F>("Parameters", "", 2, 0.0, 2.0, 8, 0, 8.);
      auto h2_BSShadow =
          std::make_unique<TH2F>("Shadow", ";;;#Delta parameter (payload A - payload B)", 2, 0.0, 2.0, 8, 0, 8.);
      h2_BSParameters->SetStats(false);
      h2_BSShadow->SetStats(false);

      std::function<double(parameters, bool)> cutFunctor = [this](parameters my_param, bool isError) {
        double ret(-999.);
        if (!isError) {
          switch (my_param) {
            case X:
              return (f_payload->x() - l_payload->x());
            case Y:
              return (f_payload->y() - l_payload->y());
            case Z:
              return (f_payload->z() - l_payload->z());
            case sigmaX:
              return (f_payload->beamWidthX() - l_payload->beamWidthX());
            case sigmaY:
              return (f_payload->beamWidthY() - l_payload->beamWidthY());
            case sigmaZ:
              return (f_payload->sigmaZ() - l_payload->sigmaZ());
            case dxdz:
              return (f_payload->dxdz() - l_payload->dxdz());
            case dydz:
              return (f_payload->dydz() - l_payload->dydz());
            case END_OF_TYPES:
              return ret;
            default:
              return ret;
          }
        } else {
          switch (my_param) {
            case X:
              return (f_payload->xError() - l_payload->xError());
            case Y:
              return (f_payload->yError() - l_payload->yError());
            case Z:
              return (f_payload->zError() - l_payload->zError());
            case sigmaX:
              return (f_payload->beamWidthXError() - l_payload->beamWidthXError());
            case sigmaY:
              return (f_payload->beamWidthYError() - l_payload->beamWidthYError());
            case sigmaZ:
              return (f_payload->sigmaZError() - l_payload->sigmaZError());
            case dxdz:
              return (f_payload->dxdzError() - l_payload->dxdzError());
            case dydz:
              return (f_payload->dydzError() - l_payload->dydzError());
            case END_OF_TYPES:
              return ret;
            default:
              return ret;
          }
        }
      };

      h2_BSParameters->GetXaxis()->SetBinLabel(1, "Value");
      h2_BSParameters->GetXaxis()->SetBinLabel(2, "Error");
      h2_BSShadow->GetXaxis()->SetBinLabel(1, "Value");
      h2_BSShadow->GetXaxis()->SetBinLabel(2, "Error");

      unsigned int yBin = 8;
      for (int foo = parameters::X; foo <= parameters::dydz; foo++) {
        parameters param = static_cast<parameters>(foo);
        std::string theLabel = getStringFromTypeEnum(param);
        h2_BSParameters->GetYaxis()->SetBinLabel(yBin, theLabel.c_str());
        h2_BSParameters->SetBinContent(1, yBin, cutFunctor(param, false));
        h2_BSParameters->SetBinContent(2, yBin, cutFunctor(param, true));
        h2_BSShadow->GetYaxis()->SetBinLabel(yBin, theLabel.c_str());
        h2_BSShadow->SetBinContent(1, yBin, cutFunctor(param, false));
        h2_BSShadow->SetBinContent(2, yBin, cutFunctor(param, true));
        yBin--;
      }

      h2_BSParameters->GetXaxis()->LabelsOption("h");
      h2_BSParameters->GetYaxis()->SetLabelSize(0.05);
      h2_BSParameters->GetXaxis()->SetLabelSize(0.05);
      h2_BSShadow->GetXaxis()->LabelsOption("h");
      h2_BSShadow->GetYaxis()->SetLabelSize(0.05);
      h2_BSShadow->GetXaxis()->SetLabelSize(0.05);
      h2_BSShadow->GetZaxis()->CenterTitle();
      h2_BSShadow->GetZaxis()->SetTitleOffset(1.5);
      h2_BSParameters->SetMarkerSize(1.5);

      // this is the fine gradient palette (blue to red)
      double max = h2_BSShadow->GetMaximum();
      double min = h2_BSShadow->GetMinimum();
      double val_white = 0.;
      double per_white = (max != min) ? ((val_white - min) / (max - min)) : 0.5;

      const int Number = 3;
      double Red[Number] = {0., 1., 1.};
      double Green[Number] = {0., 1., 0.};
      double Blue[Number] = {1., 1., 0.};
      double Stops[Number] = {0., per_white, 1.};
      int nb = 256;
      h2_BSShadow->SetContour(nb);
      TColor::CreateGradientColorTable(Number, Stops, Red, Green, Blue, nb);

      h2_BSShadow->Draw("colz");
      h2_BSParameters->Draw("TEXTsame");

      auto ltx = TLatex();
      ltx.SetTextFont(62);
      ltx.SetTextSize(0.025);
      ltx.SetTextAlign(11);

      auto l_runLS = BeamSpotPI::unpack(std::get<0>(lastiov));
      auto f_runLS = BeamSpotPI::unpack(std::get<0>(firstiov));

      if (this->m_plotAnnotations.ntags == 2) {
        ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                         1 - gPad->GetTopMargin() + 0.03,
                         ("#splitline{A = #color[4]{" + f_tagname + "}}{B = #color[4]{" + l_tagname + "}}").c_str());
      } else {
        ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                         1 - gPad->GetTopMargin() + 0.03,
                         ("#splitline{#color[4]{" + f_tagname + "}}{A = " + std::to_string(l_runLS.first) + "," +
                          std::to_string(l_runLS.second) + " B =" + std::to_string(f_runLS.first) + "," +
                          std::to_string(f_runLS.second) + "}")
                             .c_str());
      }

      std::string fileName(this->m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

  public:
    virtual std::shared_ptr<TH2F> fillTheExtraHistogram() const { return nullptr; }

  protected:
    bool isOnline_;
    std::shared_ptr<PayloadType> f_payload;
    std::shared_ptr<PayloadType> l_payload;

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
