#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"

#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CondCore/SiStripPlugins/interface/SiStripPayloadInspectorHelper.h"
#include "CalibTracker/SiStripCommon/interface/StandaloneTrackerTopology.h" 

#include <memory>
#include <sstream>

// include ROOT 
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"
#include "TGaxis.h"

namespace {

  /************************************************
    TrackerMap of SiStrip Lorentz Angle
  *************************************************/
  class SiStripLorentzAngle_TrackerMap : public cond::payloadInspector::PlotImage<SiStripLorentzAngle> {
  public:
    SiStripLorentzAngle_TrackerMap() : cond::payloadInspector::PlotImage<SiStripLorentzAngle>( "Tracker Map IsModuleLVOff" ){
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<SiStripLorentzAngle> payload = fetchPayload( std::get<1>(iov) );

      std::unique_ptr<TrackerMap> tmap = std::unique_ptr<TrackerMap>(new TrackerMap("SiStripLorentzAngle"));
      tmap->setPalette(1);
      std::string titleMap = "TrackerMap of SiStrip Lorentz Angle per module, payload : "+std::get<1>(iov);
      tmap->setTitle(titleMap);

      std::map<uint32_t,float> LAMap_ = payload->getLorentzAngles();
      
      for(const auto &element : LAMap_){
	tmap->fill(element.first,element.second);
      } // loop over the LA MAP
      
      std::string fileName(m_imageFileName);
      tmap->save(true,0,0,fileName);

      return true;
    }
  };

}

PAYLOAD_INSPECTOR_MODULE( SiStripLorentaAngle ){
  PAYLOAD_INSPECTOR_CLASS( SiStripLorentzAngle_TrackerMap );
}
