#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/BeamSpotPlugins/interface/BeamSpotPayloadInspectorHelper.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"

#include <TStyle.h>

namespace {

  using namespace BeamSpotPI;

  /************************************************
    history plots
  *************************************************/
  typedef BeamSpot_history<X, BeamSpotOnlineObjects> BeamSpotOnline_HistoryX;
  typedef BeamSpot_history<Y, BeamSpotOnlineObjects> BeamSpotOnline_HistoryY;
  typedef BeamSpot_history<Z, BeamSpotOnlineObjects> BeamSpotOnline_HistoryZ;
  typedef BeamSpot_history<sigmaX, BeamSpotOnlineObjects> BeamSpotOnline_HistorySigmaX;
  typedef BeamSpot_history<sigmaY, BeamSpotOnlineObjects> BeamSpotOnline_HistorySigmaY;
  typedef BeamSpot_history<sigmaZ, BeamSpotOnlineObjects> BeamSpotOnline_HistorySigmaZ;
  typedef BeamSpot_history<dxdz, BeamSpotOnlineObjects> BeamSpotOnline_HistorydXdZ;
  typedef BeamSpot_history<dydz, BeamSpotOnlineObjects> BeamSpotOnline_HistorydYdZ;

  /************************************************
    run history plots
  *************************************************/

  typedef BeamSpot_runhistory<X, BeamSpotOnlineObjects> BeamSpotOnline_RunHistoryX;
  typedef BeamSpot_runhistory<Y, BeamSpotOnlineObjects> BeamSpotOnline_RunHistoryY;
  typedef BeamSpot_runhistory<Z, BeamSpotOnlineObjects> BeamSpotOnline_RunHistoryZ;
  typedef BeamSpot_runhistory<sigmaX, BeamSpotOnlineObjects> BeamSpotOnline_RunHistorySigmaX;
  typedef BeamSpot_runhistory<sigmaY, BeamSpotOnlineObjects> BeamSpotOnline_RunHistorySigmaY;
  typedef BeamSpot_runhistory<sigmaZ, BeamSpotOnlineObjects> BeamSpotOnline_RunHistorySigmaZ;
  typedef BeamSpot_runhistory<dxdz, BeamSpotOnlineObjects> BeamSpotOnline_RunHistorydXdZ;
  typedef BeamSpot_runhistory<dydz, BeamSpotOnlineObjects> BeamSpotOnline_RunHistorydYdZ;

  /************************************************
    time history plots
  *************************************************/

  typedef BeamSpot_timehistory<X, BeamSpotOnlineObjects> BeamSpotOnline_TimeHistoryX;
  typedef BeamSpot_timehistory<Y, BeamSpotOnlineObjects> BeamSpotOnline_TimeHistoryY;
  typedef BeamSpot_timehistory<Z, BeamSpotOnlineObjects> BeamSpotOnline_TimeHistoryZ;
  typedef BeamSpot_timehistory<sigmaX, BeamSpotOnlineObjects> BeamSpotOnline_TimeHistorySigmaX;
  typedef BeamSpot_timehistory<sigmaY, BeamSpotOnlineObjects> BeamSpotOnline_TimeHistorySigmaY;
  typedef BeamSpot_timehistory<sigmaZ, BeamSpotOnlineObjects> BeamSpotOnline_TimeHistorySigmaZ;
  typedef BeamSpot_timehistory<dxdz, BeamSpotOnlineObjects> BeamSpotOnline_TimeHistorydXdZ;
  typedef BeamSpot_timehistory<dydz, BeamSpotOnlineObjects> BeamSpotOnline_TimeHistorydYdZ;

  /************************************************
    X-Y correlation plot
  *************************************************/

  typedef xyCorrelation<BeamSpotOnlineObjects> BeamSpotOnline_xy;

  /************************************************
    Display of Beam Spot parameters
  *************************************************/

  class BeamSpotOnlineParameters : public DisplayParameters<BeamSpotOnlineObjects> {
  public:
    std::shared_ptr<TH2F> fillTheExtraHistogram() const override {
      gStyle->SetHistMinimumZero();
      auto h2_ExtraBSParameters = std::make_shared<TH2F>("ExtraParameters", "", 1, 0.0, 1.0, 6, 0, 6.);
      h2_ExtraBSParameters->SetStats(false);

      std::function<int(parameters)> mycutFunctor = [this](parameters my_param) {
        int ret(-999.);
        switch (my_param) {
          case lastLumi:
            return this->m_payload->GetLastAnalyzedLumi();
          case lastRun:
            return this->m_payload->GetLastAnalyzedRun();
          case lastFill:
            return this->m_payload->GetLastAnalyzedFill();
          case nTracks:
            return this->m_payload->GetNumTracks();
          case nPVs:
            return this->m_payload->GetNumPVs();
          case END_OF_TYPES:
            return ret;
          default:
            return ret;
        }
      };

      unsigned int yBin = 6;
      for (int foo = parameters::lastLumi; foo != parameters::END_OF_TYPES; foo++) {
        parameters param = static_cast<parameters>(foo);
        std::string theLabel = this->getStringFromTypeEnum(param);
        h2_ExtraBSParameters->GetYaxis()->SetBinLabel(yBin, theLabel.c_str());

        edm::LogInfo("BeamSpotOnline_PayloadInspector")
            << theLabel.c_str() << " : " << mycutFunctor(param) << std::endl;

        if (param == BeamSpotPI::creationTime) {
          h2_ExtraBSParameters->SetBinContent(1, yBin, m_payload->GetCreationTime());
        } else {
          h2_ExtraBSParameters->SetBinContent(1, yBin, mycutFunctor(param));
        }

        yBin--;
      }

      h2_ExtraBSParameters->GetYaxis()->SetLabelSize(0.05);
      h2_ExtraBSParameters->GetXaxis()->SetLabelSize(0.05);
      h2_ExtraBSParameters->GetXaxis()->SetBinLabel(1, "Value");
      h2_ExtraBSParameters->SetMarkerSize(2.);

      return h2_ExtraBSParameters;
    }

  protected:
    /************************************************/
    std::string getStringFromTypeEnum(const parameters& parameter) const override {
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
        case lastLumi:
          return "last LS";
        case lastRun:
          return "last Run";
        case lastFill:
          return "last Fill";
        case nTracks:
          return "# tracks";
        case nPVs:
          return "# PVs";
        case creationTime:
          return "time";
        default:
          return "should never be here";
      }
    }
  };

}  // namespace

PAYLOAD_INSPECTOR_MODULE(BeamSpotOnline) {
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_xy);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnlineParameters);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_HistoryX);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_HistoryY);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_HistoryZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_HistorySigmaX);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_HistorySigmaY);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_HistorySigmaZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_HistorydXdZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_HistorydYdZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_RunHistoryX);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_RunHistoryY);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_RunHistoryZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_RunHistorySigmaX);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_RunHistorySigmaY);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_RunHistorySigmaZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_RunHistorydXdZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_RunHistorydYdZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_TimeHistoryX);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_TimeHistoryY);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_TimeHistoryZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_TimeHistorySigmaX);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_TimeHistorySigmaY);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_TimeHistorySigmaZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_TimeHistorydXdZ);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_TimeHistorydYdZ);
}
