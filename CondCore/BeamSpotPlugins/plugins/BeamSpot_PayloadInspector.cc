#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

#include <memory>
#include <sstream>
#include "TCanvas.h"
#include "TH2F.h"

namespace {

  enum parameters { X, Y, Z, sigmaX, sigmaY, sigmaZ, dxdz, dydz, END_OF_TYPES };

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

  class BeamSpot_hx : public cond::payloadInspector::HistoryPlot<BeamSpotObjects, std::pair<double, double> > {
  public:
    BeamSpot_hx()
        : cond::payloadInspector::HistoryPlot<BeamSpotObjects, std::pair<double, double> >("x vs run number", "x") {}

    std::pair<double, double> getFromPayload(BeamSpotObjects& payload) override {
      return std::make_pair(payload.GetX(), payload.GetXError());
    }
  };

  class BeamSpot_rhx : public cond::payloadInspector::RunHistoryPlot<BeamSpotObjects, std::pair<double, double> > {
  public:
    BeamSpot_rhx()
        : cond::payloadInspector::RunHistoryPlot<BeamSpotObjects, std::pair<double, double> >("x vs run number", "x") {}

    std::pair<double, double> getFromPayload(BeamSpotObjects& payload) override {
      return std::make_pair(payload.GetX(), payload.GetXError());
    }
  };
  class BeamSpot_x : public cond::payloadInspector::TimeHistoryPlot<BeamSpotObjects, std::pair<double, double> > {
  public:
    BeamSpot_x()
        : cond::payloadInspector::TimeHistoryPlot<BeamSpotObjects, std::pair<double, double> >("x vs time", "x") {}

    std::pair<double, double> getFromPayload(BeamSpotObjects& payload) override {
      return std::make_pair(payload.GetX(), payload.GetXError());
    }
  };

  class BeamSpot_y : public cond::payloadInspector::TimeHistoryPlot<BeamSpotObjects, std::pair<double, double> > {
  public:
    BeamSpot_y()
        : cond::payloadInspector::TimeHistoryPlot<BeamSpotObjects, std::pair<double, double> >("y vs time", "y") {}

    std::pair<double, double> getFromPayload(BeamSpotObjects& payload) override {
      return std::make_pair(payload.GetY(), payload.GetYError());
    }
  };

  /************************************************
    template classes (history)
  *************************************************/

  template <parameters my_param>
  class BeamSpot_history : public cond::payloadInspector::HistoryPlot<BeamSpotObjects, std::pair<double, double> > {
  public:
    BeamSpot_history()
        : cond::payloadInspector::HistoryPlot<BeamSpotObjects, std::pair<double, double> >(
              getStringFromParamEnum(my_param) + " vs run number", getStringFromParamEnum(my_param)) {}

    std::pair<double, double> getFromPayload(BeamSpotObjects& payload) override {
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

  typedef BeamSpot_history<X> BeamSpot_HistoryX;
  typedef BeamSpot_history<Y> BeamSpot_HistoryY;
  typedef BeamSpot_history<Z> BeamSpot_HistoryZ;
  typedef BeamSpot_history<sigmaX> BeamSpot_HistorySigmaX;
  typedef BeamSpot_history<sigmaY> BeamSpot_HistorySigmaY;
  typedef BeamSpot_history<sigmaZ> BeamSpot_HistorySigmaZ;
  typedef BeamSpot_history<dxdz> BeamSpot_HistorydXdZ;
  typedef BeamSpot_history<dydz> BeamSpot_HistorydYdZ;

  /************************************************
    template classes (run history)
  *************************************************/

  template <parameters my_param>
  class BeamSpot_runhistory
      : public cond::payloadInspector::RunHistoryPlot<BeamSpotObjects, std::pair<double, double> > {
  public:
    BeamSpot_runhistory()
        : cond::payloadInspector::RunHistoryPlot<BeamSpotObjects, std::pair<double, double> >(
              getStringFromParamEnum(my_param) + " vs run number", getStringFromParamEnum(my_param)) {}

    std::pair<double, double> getFromPayload(BeamSpotObjects& payload) override {
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

  typedef BeamSpot_runhistory<X> BeamSpot_RunHistoryX;
  typedef BeamSpot_runhistory<Y> BeamSpot_RunHistoryY;
  typedef BeamSpot_runhistory<Z> BeamSpot_RunHistoryZ;
  typedef BeamSpot_runhistory<sigmaX> BeamSpot_RunHistorySigmaX;
  typedef BeamSpot_runhistory<sigmaY> BeamSpot_RunHistorySigmaY;
  typedef BeamSpot_runhistory<sigmaZ> BeamSpot_RunHistorySigmaZ;
  typedef BeamSpot_runhistory<dxdz> BeamSpot_RunHistorydXdZ;
  typedef BeamSpot_runhistory<dydz> BeamSpot_RunHistorydYdZ;

  /************************************************
    template classes (time history)
  *************************************************/

  template <parameters my_param>
  class BeamSpot_timehistory
      : public cond::payloadInspector::TimeHistoryPlot<BeamSpotObjects, std::pair<double, double> > {
  public:
    BeamSpot_timehistory()
        : cond::payloadInspector::TimeHistoryPlot<BeamSpotObjects, std::pair<double, double> >(
              getStringFromParamEnum(my_param) + " vs time", getStringFromParamEnum(my_param)) {}

    std::pair<double, double> getFromPayload(BeamSpotObjects& payload) override {
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

  typedef BeamSpot_timehistory<X> BeamSpot_TimeHistoryX;
  typedef BeamSpot_timehistory<Y> BeamSpot_TimeHistoryY;
  typedef BeamSpot_timehistory<Z> BeamSpot_TimeHistoryZ;
  typedef BeamSpot_timehistory<sigmaX> BeamSpot_TimeHistorySigmaX;
  typedef BeamSpot_timehistory<sigmaY> BeamSpot_TimeHistorySigmaY;
  typedef BeamSpot_timehistory<sigmaZ> BeamSpot_TimeHistorySigmaZ;
  typedef BeamSpot_timehistory<dxdz> BeamSpot_TimeHistorydXdZ;
  typedef BeamSpot_timehistory<dydz> BeamSpot_TimeHistorydYdZ;

  /************************************************
    X-Y correlation plot
  *************************************************/
  class BeamSpot_xy : public cond::payloadInspector::ScatterPlot<BeamSpotObjects, double, double> {
  public:
    BeamSpot_xy() : cond::payloadInspector::ScatterPlot<BeamSpotObjects, double, double>("BeamSpot x vs y", "x", "y") {}

    std::tuple<double, double> getFromPayload(BeamSpotObjects& payload) override {
      return std::make_tuple(payload.GetX(), payload.GetY());
    }
  };

  /************************************************
    Display of Beam Spot parameters
  *************************************************/
  class BeamSpotParameters : public cond::payloadInspector::PlotImage<BeamSpotObjects> {
  public:
    BeamSpotParameters() : cond::payloadInspector::PlotImage<BeamSpotObjects>("Display of BeamSpot parameters") {
      setSingleIov(true);
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
      auto iov = iovs.front();
      std::shared_ptr<BeamSpotObjects> payload = fetchPayload(std::get<1>(iov));

      TCanvas canvas("Beam Spot Parameters Summary", "BeamSpot Parameters summary", 1000, 1000);
      canvas.cd();

      canvas.SetTopMargin(0.07);
      canvas.SetBottomMargin(0.06);
      canvas.SetLeftMargin(0.15);
      canvas.SetRightMargin(0.03);
      canvas.Modified();
      canvas.SetGrid();

      auto h2_BSParameters =
          std::make_unique<TH2F>("Parameters", "BeamSpot parameters summary", 2, 0.0, 2.0, 8, 0, 8.);
      h2_BSParameters->SetStats(false);

      std::function<double(parameters, bool)> cutFunctor = [&payload](parameters my_param, bool isError) {
        double ret(-999.);
        if (!isError) {
          switch (my_param) {
            case X:
              return payload->GetX();
            case Y:
              return payload->GetY();
            case Z:
              return payload->GetZ();
            case sigmaX:
              return payload->GetBeamWidthX();
            case sigmaY:
              return payload->GetBeamWidthY();
            case sigmaZ:
              return payload->GetSigmaZ();
            case dxdz:
              return payload->Getdxdz();
            case dydz:
              return payload->Getdydz();
            case END_OF_TYPES:
              return ret;
            default:
              return ret;
          }
        } else {
          switch (my_param) {
            case X:
              return payload->GetXError();
            case Y:
              return payload->GetYError();
            case Z:
              return payload->GetZError();
            case sigmaX:
              return payload->GetBeamWidthXError();
            case sigmaY:
              return payload->GetBeamWidthYError();
            case sigmaZ:
              return payload->GetSigmaZError();
            case dxdz:
              return payload->GetdxdzError();
            case dydz:
              return payload->GetdydzError();
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
      for (int foo = parameters::X; foo != parameters::END_OF_TYPES; foo++) {
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

      std::string fileName(m_imageFileName);
      canvas.SaveAs(fileName.c_str());

      return true;
    }

    /************************************************/
    std::string getStringFromTypeEnum(const parameters& parameter) {
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

}  // namespace

PAYLOAD_INSPECTOR_MODULE(BeamSpot) {
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_hx);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_rhx);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_x);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_y);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_xy);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotParameters);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_HistoryX);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_HistoryY);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_HistoryZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_HistorySigmaX);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_HistorySigmaY);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_HistorySigmaZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_HistorydXdZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_HistorydYdZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_RunHistoryX);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_RunHistoryY);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_RunHistoryZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_RunHistorySigmaX);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_RunHistorySigmaY);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_RunHistorySigmaZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_RunHistorydXdZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_RunHistorydYdZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_TimeHistoryX);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_TimeHistoryY);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_TimeHistoryZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_TimeHistorySigmaX);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_TimeHistorySigmaY);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_TimeHistorySigmaZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_TimeHistorydXdZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_TimeHistorydYdZ);
}
