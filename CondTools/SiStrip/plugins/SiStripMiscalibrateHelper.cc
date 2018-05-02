#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"
#include "CommonTools/TrackerMap/interface/TmModule.h"
#include "CondTools/SiStrip/interface/SiStripMiscalibrateHelper.h"
#include "CondCore/SiStripPlugins/interface/SiStripPayloadInspectorHelper.h"

/*--------------------------------------------------------------------*/
sistripsummary::TrackerRegion SiStripMiscalibrate::getRegionFromString(std::string region)
/*--------------------------------------------------------------------*/
{
  std::map<std::string,sistripsummary::TrackerRegion> mapping = {
    {"Tracker",sistripsummary::TRACKER},
    {"TIB"    ,sistripsummary::TIB   },
    {"TIB_1"  ,sistripsummary::TIB_1 },
    {"TIB_2"  ,sistripsummary::TIB_2 },
    {"TIB_3"  ,sistripsummary::TIB_3 },
    {"TIB_4"  ,sistripsummary::TIB_4 },
    {"TID"    ,sistripsummary::TID   },
    {"TIDP"   ,sistripsummary::TIDP  },
    {"TIDP_1" ,sistripsummary::TIDP_1},
    {"TIDP_2" ,sistripsummary::TIDP_2},
    {"TIDP_3" ,sistripsummary::TIDP_3},
    {"TIDM"   ,sistripsummary::TIDM  },
    {"TIDM_1" ,sistripsummary::TIDM_1},
    {"TIDM_2" ,sistripsummary::TIDM_2},
    {"TIDM_3" ,sistripsummary::TIDM_3},
    {"TOB"    ,sistripsummary::TOB   },
    {"TOB_1"  ,sistripsummary::TOB_1 },
    {"TOB_2"  ,sistripsummary::TOB_2 },
    {"TOB_3"  ,sistripsummary::TOB_3 },
    {"TOB_4"  ,sistripsummary::TOB_4 },
    {"TOB_5"  ,sistripsummary::TOB_5 },
    {"TOB_6"  ,sistripsummary::TOB_6 },
    {"TEC"    ,sistripsummary::TEC   },
    {"TECP"   ,sistripsummary::TECP  },
    {"TECP_1" ,sistripsummary::TECP_1},
    {"TECP_2" ,sistripsummary::TECP_2},
    {"TECP_3" ,sistripsummary::TECP_3},
    {"TECP_4" ,sistripsummary::TECP_4},
    {"TECP_5" ,sistripsummary::TECP_5},
    {"TECP_6" ,sistripsummary::TECP_6},
    {"TECP_7" ,sistripsummary::TECP_7},
    {"TECP_8" ,sistripsummary::TECP_8},
    {"TECP_9" ,sistripsummary::TECP_9},
    {"TECM"   ,sistripsummary::TECM  },
    {"TECM_1" ,sistripsummary::TECM_1},
    {"TECM_2" ,sistripsummary::TECM_2},
    {"TECM_3" ,sistripsummary::TECM_3},
    {"TECM_4" ,sistripsummary::TECM_4},
    {"TECM_5" ,sistripsummary::TECM_5},
    {"TECM_6" ,sistripsummary::TECM_6},
    {"TECM_7" ,sistripsummary::TECM_7},
    {"TECM_8" ,sistripsummary::TECM_8},
    {"TECM_9" ,sistripsummary::TECM_9}
  };
  
  if (mapping.find(region) == mapping.end() ){
    edm::LogError("SiStripMiscalibrate") << "@SUB=analyze" << "Unknown partition: " << region;
    throw cms::Exception("Invalid Partition passed"); 
  } else {
    return mapping[region];
  }
}

/*--------------------------------------------------------------------*/
std::vector<sistripsummary::TrackerRegion> SiStripMiscalibrate::getRegionsFromDetId(const TrackerTopology* m_trackerTopo,DetId detid)
/*--------------------------------------------------------------------*/      
{
  int layer    = 0;
  int side     = 0;
  int subdet   = 0;
  int detCode  = 0;

  std::vector<sistripsummary::TrackerRegion> ret;

  switch (detid.subdetId()) {
  case StripSubdetector::TIB:
    layer = m_trackerTopo->tibLayer(detid);
    subdet = 1;
    break;
  case StripSubdetector::TOB:
    layer = m_trackerTopo->tobLayer(detid);
    subdet = 2;
    break;
  case StripSubdetector::TID:
    // is this module in TID+ or TID-?
    layer = m_trackerTopo->tidWheel(detid);
    side  = m_trackerTopo->tidSide(detid);
    subdet = 3*10+side;
    break;
  case StripSubdetector::TEC:
    // is this module in TEC+ or TEC-?
    layer = m_trackerTopo->tecWheel(detid);
    side  = m_trackerTopo->tecSide(detid);
    subdet = 4*10+side;
    break;
  }
  
  detCode = (subdet*10)+layer;
  
  ret.push_back(static_cast<sistripsummary::TrackerRegion>(detCode));

  if(subdet/10 > 0) {
    ret.push_back(static_cast<sistripsummary::TrackerRegion>(subdet/10));
  }

  ret.push_back(static_cast<sistripsummary::TrackerRegion>(subdet));
  ret.push_back(sistripsummary::TRACKER);

  return ret;
}

/*--------------------------------------------------------------------*/
std::pair<float,float> SiStripMiscalibrate::getTruncatedRange(const TrackerMap* theMap){
/*--------------------------------------------------------------------*/  

  // ------------ trim the tracker map  ------------

  auto map = theMap->smoduleMap;
  std::map<unsigned int,float> info_per_detid;
  for (int layer=1; layer < 44; layer++){
    for (int ring=theMap->firstRing[layer-1]; ring < theMap->ntotRing[layer-1]+theMap->firstRing[layer-1];ring++){
      for (int module=1;module<200;module++) {
	int key=layer*100000+ring*1000+module;
	TmModule* mod = map[key];
	if(mod !=nullptr && !mod->notInUse()  && mod->count>0){
	  info_per_detid[key]=mod->value;
	}
      } // loop on modules
    } // loop on ring
  } // loop on layers
  
  auto range = SiStripPI::getTheRange(info_per_detid,2);
  return range;
  
}
