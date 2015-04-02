#ifndef Phase2OuterTracker_OuterTrackerMonitorPixelDigiMaps_h
#define Phase2OuterTracker_OuterTrackerMonitorPixelDigiMaps_h

#include <vector>
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TH2D.h>

class DQMStore;

class OuterTrackerMonitorPixelDigiMaps : public edm::EDAnalyzer {

public:
  explicit OuterTrackerMonitorPixelDigiMaps(const edm::ParameterSet&);
  ~OuterTrackerMonitorPixelDigiMaps();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
 
	
  MonitorElement* PixelDigiMaps_Barrel_XY = 0;
  MonitorElement* PixelDigiMaps_Barrel_XY_Zoom = 0;
  MonitorElement* PixelDigiMaps_Endcap_Fw_XY = 0;
  MonitorElement* PixelDigiMaps_Endcap_Bw_XY = 0;
  MonitorElement* PixelDigiMaps_RZ = 0;
  MonitorElement* PixelDigiMaps_Endcap_Fw_RZ_Zoom = 0;
  MonitorElement* PixelDigiMaps_Endcap_Bw_RZ_Zoom = 0;
  
  std::map< std::string, TH2D* > m_hPixelDigi_Barrel_XY_Survey;
  std::map< std::string, TH2D* > m_hPixelDigi_RZ_S;

 private:
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;

  std::string topFolderName_;
};
#endif
