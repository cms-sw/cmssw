#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"

#include "CommonTools/TrackerMap/interface/TrackerMap.h"

#include <memory>
#include <sstream>

namespace {

  class SiStripDetVOff_LV : public cond::payloadInspector::TimeHistoryPlot<SiStripDetVOff,int>{
  public:
    SiStripDetVOff_LV(): cond::payloadInspector::TimeHistoryPlot<SiStripDetVOff,int >( "Nr of mod with LV OFF vs time", "nLVOff"){
    }

    int getFromPayload( SiStripDetVOff& payload ) override{
      return payload.getLVoffCounts();
    }

  };

  class SiStripDetVOff_HV : public cond::payloadInspector::TimeHistoryPlot<SiStripDetVOff,int> {
  public:
    SiStripDetVOff_HV() : cond::payloadInspector::TimeHistoryPlot<SiStripDetVOff,int >( "Nr of mod with HV OFF vs time","nHVOff"){
    }

    int getFromPayload( SiStripDetVOff& payload ) override{
      return payload.getHVoffCounts();
    }

  };

  /************************************************
    TrackerMap of Module VOff
  *************************************************/
  class SiStripDetVOff_IsModuleVOff_TrackerMap : public cond::payloadInspector::PlotImage<SiStripDetVOff> {
  public:
    SiStripDetVOff_IsModuleVOff_TrackerMap() : cond::payloadInspector::PlotImage<SiStripDetVOff>( "Tracker Map IsModuleVOff" ){
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripDetVOff> payload = fetchPayload( std::get<1>(iov) );

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripIsModuleVOff"));
      tmap->setPalette(1);
      std::string titleMap = "TrackerMap of VOff modules (HV or LV), payload : "+std::get<1>(iov);
      tmap->setTitle(titleMap);

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      for (const auto & d : detid) {
	if(payload->IsModuleVOff(d)){
	  tmap->fill(d,1);
	}
      } // loop over detIds
      
      std::string fileName(m_imageFileName);
      tmap->save(true,0,0,fileName);

      return true;
    }
  };

  /************************************************
    TrackerMap of Module HVOff
  *************************************************/
  class SiStripDetVOff_IsModuleHVOff_TrackerMap : public cond::payloadInspector::PlotImage<SiStripDetVOff> {
  public:
    SiStripDetVOff_IsModuleHVOff_TrackerMap() : cond::payloadInspector::PlotImage<SiStripDetVOff>( "Tracker Map IsModuleHVOff" ){
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripDetVOff> payload = fetchPayload( std::get<1>(iov) );

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripIsModuleHVOff"));
      tmap->setPalette(1);
      std::string titleMap = "TrackerMap of HV Off modules, payload : "+std::get<1>(iov);
      tmap->setTitle(titleMap);

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      for (const auto & d : detid) {
	if(payload->IsModuleHVOff(d)){
	  tmap->fill(d,1);
	}
      } // loop over detIds
      
      std::string fileName(m_imageFileName);
      tmap->save(true,0,0,fileName);

      return true;
    }
  };

  /************************************************
    TrackerMap of Module LVOff
  *************************************************/
  class SiStripDetVOff_IsModuleLVOff_TrackerMap : public cond::payloadInspector::PlotImage<SiStripDetVOff> {
  public:
    SiStripDetVOff_IsModuleLVOff_TrackerMap() : cond::payloadInspector::PlotImage<SiStripDetVOff>( "Tracker Map IsModuleLVOff" ){
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripDetVOff> payload = fetchPayload( std::get<1>(iov) );

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripIsModuleLVOff"));
      tmap->setPalette(1);
      std::string titleMap = "TrackerMap of LV Off modules, payload : "+std::get<1>(iov);
      tmap->setTitle(titleMap);

      std::vector<uint32_t> detid;
      payload->getDetIds(detid);

      for (const auto & d : detid) {
	if(payload->IsModuleLVOff(d)){
	  tmap->fill(d,1);
	}
      } // loop over detIds
      
      std::string fileName(m_imageFileName);
      tmap->save(true,0,0,fileName);

      return true;
    }
  };

}

PAYLOAD_INSPECTOR_MODULE( SiStripDetVOff ){
  PAYLOAD_INSPECTOR_CLASS( SiStripDetVOff_LV );
  PAYLOAD_INSPECTOR_CLASS( SiStripDetVOff_HV );
  PAYLOAD_INSPECTOR_CLASS( SiStripDetVOff_IsModuleVOff_TrackerMap );
  PAYLOAD_INSPECTOR_CLASS( SiStripDetVOff_IsModuleLVOff_TrackerMap );
  PAYLOAD_INSPECTOR_CLASS( SiStripDetVOff_IsModuleHVOff_TrackerMap );
}
