#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

#include <memory>
#include <sstream>

namespace {

  class BeamSpot_hx : public cond::payloadInspector::HistoryPlot<BeamSpotObjects,std::pair<double,double> > {
  public:
    BeamSpot_hx(): cond::payloadInspector::HistoryPlot<BeamSpotObjects,std::pair<double,double> >( "x vs run number", "x"){
    }

    std::pair<double,double> getFromPayload( BeamSpotObjects& payload ) override{
      return std::make_pair(payload.GetX(),payload.GetXError());
    }
  };

  class BeamSpot_rhx : public cond::payloadInspector::RunHistoryPlot<BeamSpotObjects,std::pair<double,double> > {
  public:
    BeamSpot_rhx(): cond::payloadInspector::RunHistoryPlot<BeamSpotObjects,std::pair<double,double> >( "x vs run number", "x"){
    }

    std::pair<double,double> getFromPayload( BeamSpotObjects& payload ) override{
      return std::make_pair(payload.GetX(),payload.GetXError());
    }
  };
  class BeamSpot_x : public cond::payloadInspector::TimeHistoryPlot<BeamSpotObjects,std::pair<double,double> > {
  public:
    BeamSpot_x(): cond::payloadInspector::TimeHistoryPlot<BeamSpotObjects,std::pair<double,double> >( "x vs time", "x"){
    }

    std::pair<double,double> getFromPayload( BeamSpotObjects& payload ) override{
      return std::make_pair(payload.GetX(),payload.GetXError());
    }
  };

  class BeamSpot_y : public cond::payloadInspector::TimeHistoryPlot<BeamSpotObjects,std::pair<double,double> >{
  public:
    BeamSpot_y(): cond::payloadInspector::TimeHistoryPlot<BeamSpotObjects,std::pair<double,double> >( "y vs time", "y"){
    }

    std::pair<double,double> getFromPayload( BeamSpotObjects& payload ) override{
      return std::make_pair(payload.GetY(),payload.GetYError());
    }
  };

  class BeamSpot_xy : public cond::payloadInspector::ScatterPlot<BeamSpotObjects,double,double>{
  public:
    BeamSpot_xy(): cond::payloadInspector::ScatterPlot<BeamSpotObjects,double,double>("BeamSpot x vs y","x","y" ){
    }

    std::tuple<double,double> getFromPayload( BeamSpotObjects& payload ) override{
      return std::make_tuple( payload.GetX(), payload.GetY() );
    }
  };


}

PAYLOAD_INSPECTOR_MODULE( BeamSpot ){
  PAYLOAD_INSPECTOR_CLASS( BeamSpot_hx );
  PAYLOAD_INSPECTOR_CLASS( BeamSpot_rhx );
  PAYLOAD_INSPECTOR_CLASS( BeamSpot_x );
  PAYLOAD_INSPECTOR_CLASS( BeamSpot_y );
  PAYLOAD_INSPECTOR_CLASS( BeamSpot_xy );
}
