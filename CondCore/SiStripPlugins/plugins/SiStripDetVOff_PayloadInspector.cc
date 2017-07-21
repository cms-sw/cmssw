#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"

#include <memory>
#include <sstream>

namespace {

  class SiStripDetVOff_LV : public cond::payloadInspector::TimeHistoryPlot<SiStripDetVOff,int>{
  public:
    SiStripDetVOff_LV(): cond::payloadInspector::TimeHistoryPlot<SiStripDetVOff,int >( "Nr of mod with LV OFF vs time", "nLVOff"){
    }

    int getFromPayload( SiStripDetVOff& payload ){
      return payload.getLVoffCounts();
    }

  };

  class SiStripDetVOff_HV : public cond::payloadInspector::TimeHistoryPlot<SiStripDetVOff,int> {
  public:
    SiStripDetVOff_HV() : cond::payloadInspector::TimeHistoryPlot<SiStripDetVOff,int >( "Nr of mod with HV OFF vs time","nHVOff"){
    }

    int getFromPayload( SiStripDetVOff& payload ){
      return payload.getHVoffCounts();
    }

  };

}

PAYLOAD_INSPECTOR_MODULE( SiStripDetVOff ){
  PAYLOAD_INSPECTOR_CLASS( SiStripDetVOff_LV );
  PAYLOAD_INSPECTOR_CLASS( SiStripDetVOff_HV );
}
