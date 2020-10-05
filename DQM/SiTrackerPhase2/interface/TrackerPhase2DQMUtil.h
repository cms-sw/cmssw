#ifndef _Validation_SiTrackerPhase2V_Phase2TrackierValidationUtil_h
#define _Validation_SiTrackerPhase2V_Phase2TrackierValidationUtil_h
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include<string>
#include<sstream>

namespace Phase2TkUtil {

std::string getITHistoId(uint32_t det_id, const TrackerTopology* tTopo) {
  std::string Disc;
  std::ostringstream fname1;
  int layer = tTopo->getITPixelLayerNumber(det_id);
  
  if(layer < 0)
    return "";
  if (layer < 100) {
    fname1 << "Barrel/";
    fname1 << "Layer" << layer;
    fname1 << "";
  } else {
    int side = tTopo->pxfSide(det_id);
    fname1 << "EndCap_Side" << side << "/";
    int disc = tTopo->pxfDisk(det_id);
    Disc = (disc < 9) ? "EPix" : "FPix";;
    fname1 << Disc << "/";
    int ring = tTopo->pxfBlade(det_id);
    fname1 << "Ring" << ring;
  }
  return fname1.str();
}

std::string getOTHistoId(uint32_t det_id, const TrackerTopology* tTopo) {
  std::string Disc;
  std::ostringstream fname1;
  int layer = tTopo->getOTLayerNumber(det_id);
  
  if(layer < 0)
    return "";
  if (layer < 100) {
    fname1 << "Barrel/";
    fname1 << "Layer" << layer;
    fname1 << "";
  } else {
    int side = tTopo->tidSide(det_id);
    fname1 << "EndCap_Side" << side << "/";
    int disc = tTopo->tidWheel(det_id);
    Disc = (disc < 3) ? "TEDD_1" : "TEDD_2";
    fname1 << Disc << "/";
    int ring = tTopo->tidRing(det_id);
    fname1 << "Ring" << ring;
  }
  return fname1.str();
}

typedef dqm::reco::MonitorElement MonitorElement;
typedef dqm::reco::DQMStore DQMStore;
MonitorElement* book1DFromPSet(const edm::ParameterSet& hpars, const std::string& hname, DQMStore::IBooker& ibooker) {
  MonitorElement* temp = nullptr;
  if(hpars.getParameter<bool>("switch")) {
    temp = ibooker.book1D(hname,
			  hname,
			  hpars.getParameter<int32_t>("NxBins"),
			  hpars.getParameter<double>("xmin"),
			  hpars.getParameter<double>("xmax"));
  }
  return temp;  
}

MonitorElement* book2DFromPSet(const edm::ParameterSet& hpars, const std::string& hname, DQMStore::IBooker& ibooker) {
  MonitorElement* temp = nullptr;
  if(hpars.getParameter<bool>("switch")) {
    temp = ibooker.book2D(hname,
			  hname,
			  hpars.getParameter<int32_t>("NxBins"),
			  hpars.getParameter<double>("xmin"),
			  hpars.getParameter<double>("xmax"),
			  hpars.getParameter<int32_t>("NyBins"),
			  hpars.getParameter<double>("ymin"),
			  hpars.getParameter<double>("ymax"));
  }
  return temp;  
}

MonitorElement* bookProfile1DFromPSet(const edm::ParameterSet& hpars, const std::string& hname, DQMStore::IBooker& ibooker) {
  MonitorElement* temp = nullptr;
  if(hpars.getParameter<bool>("switch")) {
    temp = ibooker.bookProfile(hname,
			       hname,
			       hpars.getParameter<int32_t>("NxBins"),
			       hpars.getParameter<double>("xmin"),
			       hpars.getParameter<double>("xmax"),
			       hpars.getParameter<double>("ymin"),
			       hpars.getParameter<double>("ymax"));
  }
  return temp;  
}
}
 
#endif
