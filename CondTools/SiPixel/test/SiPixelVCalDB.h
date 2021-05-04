#ifndef CalibTracker_SiPixelVCalDB_SiPixelVCalDB_h
#define CalibTracker_SiPixelVCalDB_SiPixelVCalDB_h
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelVCal.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
//#include "CondTools/SiPixel/test/SiPixelVCalPixelId.h"

class SiPixelVCalDB : public edm::EDAnalyzer {
public:
  explicit SiPixelVCalDB(const edm::ParameterSet& conf);
  explicit SiPixelVCalDB();
  virtual ~SiPixelVCalDB();
  virtual void beginJob();
  virtual void endJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  std::string recordName_;
  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters BPixParameters_;
  Parameters FPixParameters_;
};

#endif
