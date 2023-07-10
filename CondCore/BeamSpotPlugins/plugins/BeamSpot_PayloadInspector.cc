#include "CondCore/BeamSpotPlugins/interface/BeamSpotPayloadInspectorHelper.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

namespace {

  using namespace beamSpotPI;

  class BeamSpot_hx : public cond::payloadInspector::HistoryPlot<BeamSpotObjects, std::pair<double, double> > {
  public:
    BeamSpot_hx()
        : cond::payloadInspector::HistoryPlot<BeamSpotObjects, std::pair<double, double> >("x vs run number", "x") {}

    std::pair<double, double> getFromPayload(BeamSpotObjects& payload) override {
      return std::make_pair(payload.x(), payload.xError());
    }
  };

  class BeamSpot_rhx : public cond::payloadInspector::RunHistoryPlot<BeamSpotObjects, std::pair<double, double> > {
  public:
    BeamSpot_rhx()
        : cond::payloadInspector::RunHistoryPlot<BeamSpotObjects, std::pair<double, double> >("x vs run number", "x") {}

    std::pair<double, double> getFromPayload(BeamSpotObjects& payload) override {
      return std::make_pair(payload.x(), payload.xError());
    }
  };
  class BeamSpot_x : public cond::payloadInspector::TimeHistoryPlot<BeamSpotObjects, std::pair<double, double> > {
  public:
    BeamSpot_x()
        : cond::payloadInspector::TimeHistoryPlot<BeamSpotObjects, std::pair<double, double> >("x vs time", "x") {}

    std::pair<double, double> getFromPayload(BeamSpotObjects& payload) override {
      return std::make_pair(payload.x(), payload.xError());
    }
  };

  class BeamSpot_y : public cond::payloadInspector::TimeHistoryPlot<BeamSpotObjects, std::pair<double, double> > {
  public:
    BeamSpot_y()
        : cond::payloadInspector::TimeHistoryPlot<BeamSpotObjects, std::pair<double, double> >("y vs time", "y") {}

    std::pair<double, double> getFromPayload(BeamSpotObjects& payload) override {
      return std::make_pair(payload.y(), payload.yError());
    }
  };

  /************************************************
    X-Y correlation plot
  *************************************************/

  typedef xyCorrelation<BeamSpotObjects> BeamSpot_xy;

  /************************************************
    history plots
  *************************************************/
  typedef BeamSpot_history<X, BeamSpotObjects> BeamSpot_HistoryX;
  typedef BeamSpot_history<Y, BeamSpotObjects> BeamSpot_HistoryY;
  typedef BeamSpot_history<Z, BeamSpotObjects> BeamSpot_HistoryZ;
  typedef BeamSpot_history<sigmaX, BeamSpotObjects> BeamSpot_HistorySigmaX;
  typedef BeamSpot_history<sigmaY, BeamSpotObjects> BeamSpot_HistorySigmaY;
  typedef BeamSpot_history<sigmaZ, BeamSpotObjects> BeamSpot_HistorySigmaZ;
  typedef BeamSpot_history<dxdz, BeamSpotObjects> BeamSpot_HistorydXdZ;
  typedef BeamSpot_history<dydz, BeamSpotObjects> BeamSpot_HistorydYdZ;

  /************************************************
    run history plots
  *************************************************/

  typedef BeamSpot_runhistory<X, BeamSpotObjects> BeamSpot_RunHistoryX;
  typedef BeamSpot_runhistory<Y, BeamSpotObjects> BeamSpot_RunHistoryY;
  typedef BeamSpot_runhistory<Z, BeamSpotObjects> BeamSpot_RunHistoryZ;
  typedef BeamSpot_runhistory<sigmaX, BeamSpotObjects> BeamSpot_RunHistorySigmaX;
  typedef BeamSpot_runhistory<sigmaY, BeamSpotObjects> BeamSpot_RunHistorySigmaY;
  typedef BeamSpot_runhistory<sigmaZ, BeamSpotObjects> BeamSpot_RunHistorySigmaZ;
  typedef BeamSpot_runhistory<dxdz, BeamSpotObjects> BeamSpot_RunHistorydXdZ;
  typedef BeamSpot_runhistory<dydz, BeamSpotObjects> BeamSpot_RunHistorydYdZ;

  /************************************************
    time history plots
  *************************************************/

  typedef BeamSpot_timehistory<X, BeamSpotObjects> BeamSpot_TimeHistoryX;
  typedef BeamSpot_timehistory<Y, BeamSpotObjects> BeamSpot_TimeHistoryY;
  typedef BeamSpot_timehistory<Z, BeamSpotObjects> BeamSpot_TimeHistoryZ;
  typedef BeamSpot_timehistory<sigmaX, BeamSpotObjects> BeamSpot_TimeHistorySigmaX;
  typedef BeamSpot_timehistory<sigmaY, BeamSpotObjects> BeamSpot_TimeHistorySigmaY;
  typedef BeamSpot_timehistory<sigmaZ, BeamSpotObjects> BeamSpot_TimeHistorySigmaZ;
  typedef BeamSpot_timehistory<dxdz, BeamSpotObjects> BeamSpot_TimeHistorydXdZ;
  typedef BeamSpot_timehistory<dydz, BeamSpotObjects> BeamSpot_TimeHistorydYdZ;

  /************************************************
    Display of Beam Spot parameters
  *************************************************/

  typedef DisplayParameters<BeamSpotObjects> BeamSpotParameters;

  /************************************************
    Display of Beam Spot parameters Differences
  *************************************************/

  typedef DisplayParametersDiff<BeamSpotObjects, cond::payloadInspector::MULTI_IOV, 1> BeamSpotParametersDiffSingleTag;
  typedef DisplayParametersDiff<BeamSpotObjects, cond::payloadInspector::SINGLE_IOV, 2> BeamSpotParametersDiffTwoTags;

}  // namespace

PAYLOAD_INSPECTOR_MODULE(BeamSpot) {
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_hx);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_rhx);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_x);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_y);
  PAYLOAD_INSPECTOR_CLASS(BeamSpot_xy);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotParameters);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotParametersDiffSingleTag);
  PAYLOAD_INSPECTOR_CLASS(BeamSpotParametersDiffTwoTags);
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
