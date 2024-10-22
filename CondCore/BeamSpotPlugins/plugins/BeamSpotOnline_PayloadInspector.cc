#include "CondCore/BeamSpotPlugins/interface/BeamSpotPayloadInspectorHelper.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"

#include <TStyle.h>

namespace {

  using namespace beamSpotPI;

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

      const auto span = parameters::startTime - parameters::lastLumi;
      edm::LogInfo("BeamSpotOnlineParameters") << "the span is" << span;

      auto h2_ExtraBSParameters =
          std::make_shared<TH2F>("ExtraParameters", "", 1, 0.0, 1.0, span, 0, static_cast<float>(span));
      h2_ExtraBSParameters->SetStats(false);

      //_____________________________________________________________________________
      std::function<int(parameters)> myIntFunctor = [this](parameters my_param) {
        int ret(-999);
        switch (my_param) {
          case lastLumi:
            return this->m_payload->lastAnalyzedLumi();
          case lastRun:
            return this->m_payload->lastAnalyzedRun();
          case lastFill:
            return this->m_payload->lastAnalyzedFill();
          case nTracks:
            return this->m_payload->numTracks();
          case nPVs:
            return this->m_payload->numPVs();
          case nUsedEvents:
            return this->m_payload->usedEvents();
          case maxPVs:
            return this->m_payload->maxPVs();
          default:
            return ret;
        }
      };

      //_____________________________________________________________________________
      std::function<float(parameters)> myFloatFunctor = [this](parameters my_param) {
        float ret(-999.);
        switch (my_param) {
          case meanPV:
            return this->m_payload->meanPV();
          case meanErrorPV:
            return this->m_payload->meanErrorPV();
          case rmsPV:
            return this->m_payload->rmsPV();
          case rmsErrorPV:
            return this->m_payload->rmsErrorPV();
          default:
            return ret;
        }
      };

      //_____________________________________________________________________________
      std::function<std::string(parameters)> myStringFunctor = [this](parameters my_param) {
        std::string ret("");
        switch (my_param) {
          case startTime:
            return this->m_payload->startTime();
          case endTime:
            return this->m_payload->endTime();
          case lumiRange:
            return this->m_payload->lumiRange();
          default:
            return ret;
        }
      };

      //_____________________________________________________________________________
      std::function<cond::Time_t(parameters)> myTimeFunctor = [this](parameters my_param) {
        cond::Time_t ret(1);
        switch (my_param) {
          case creationTime:
            return this->m_payload->creationTime();
          case startTimeStamp:
            return this->m_payload->startTimeStamp();
          case endTimeStamp:
            return this->m_payload->endTimeStamp();
          default:
            return ret;
        }
      };

      unsigned int yBin = span;
      for (int foo = parameters::lastLumi; foo != parameters::startTime; foo++) {
        parameters param = static_cast<parameters>(foo);
        std::string theLabel = this->getStringFromTypeEnum(param);
        h2_ExtraBSParameters->GetYaxis()->SetBinLabel(yBin, theLabel.c_str());

        if (foo <= parameters::maxPVs) {
          const auto output = try_<int, std::out_of_range>(std::bind(myIntFunctor, param), print_error);
          edm::LogInfo("BeamSpotOnline_PayloadInspector") << theLabel.c_str() << " : " << output << std::endl;
          h2_ExtraBSParameters->SetBinContent(1, yBin, output);
        } else if (foo <= parameters::rmsErrorPV) {
          const auto output = try_<float, std::out_of_range>(std::bind(myFloatFunctor, param), print_error);
          edm::LogInfo("BeamSpotOnline_PayloadInspector") << theLabel.c_str() << " : " << output << std::endl;
          h2_ExtraBSParameters->SetBinContent(1, yBin, output);
        } else if (foo <= parameters::endTimeStamp) {
          const auto output = try_<cond::Time_t, std::out_of_range>(std::bind(myTimeFunctor, param), print_error);
          edm::LogInfo("BeamSpotOnline_PayloadInspector") << theLabel.c_str() << " : " << output << std::endl;
          h2_ExtraBSParameters->SetBinContent(1, yBin, output);
          //} else if( foo <=parameters::lumiRange){
          // const auto output = try_<std::string,std::out_of_range>(std::bind(myStringFunctor, param), print_error);
          //edm::LogInfo("BeamSpotOnline_PayloadInspector") << theLabel.c_str() << " : " << output << std::endl;
          //h2_ExtraBSParameters->SetBinContent(1, yBin, output);
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
        case nUsedEvents:
          return "# events";
        case maxPVs:
          return "max PVs";
        case meanPV:
          return "#LT # PV #GT";
        case meanErrorPV:
          return "#LT # PV #GT error";
        case rmsPV:
          return "RMS(# PV)";
        case rmsErrorPV:
          return "RMS(# PV) error";
        case creationTime:
          return "creation time";
        case startTimeStamp:
          return "start timestamp";
        case endTimeStamp:
          return "end timestamp";
        case startTime:
          return "startTime";
        case endTime:
          return "endTime";
        case lumiRange:
          return "lumiRange";
        default:
          return "should never be here";
      }
    }

    //slightly better error handler
    static void print_error(const std::exception& e) { edm::LogError("BeamSpotOnlineParameters") << e.what() << '\n'; }

    // method to catch exceptions
    template <typename T, class Except, class Func, class Response>
    T try_(Func f, Response r) const {
      try {
        LogDebug("BeamSpotOnlineParameters") << "I have tried" << std::endl;
        return f();
      } catch (Except& e) {
        LogDebug("BeamSpotOnlineParameters") << "I have caught!" << std::endl;
        r(e);
        return static_cast<T>(-999);
      }
    }
  };

  /************************************************
    Display of Beam Spot parameters Differences
  *************************************************/

  typedef DisplayParametersDiff<BeamSpotOnlineObjects, cond::payloadInspector::MULTI_IOV, 1>
      BeamSpotOnlineParametersDiffSingleTag;
  typedef DisplayParametersDiff<BeamSpotOnlineObjects, cond::payloadInspector::SINGLE_IOV, 2>
      BeamSpotOnlineParametersDiffTwoTags;

}  // namespace

PAYLOAD_INSPECTOR_MODULE(BeamSpotOnline) {
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnline_xy);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnlineParameters);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnlineParametersDiffSingleTag);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotOnlineParametersDiffTwoTags);
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
