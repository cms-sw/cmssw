#ifndef CalibTracker_SiPixelVCalDB_SiPixelVCalDB_h
#define CalibTracker_SiPixelVCalDB_SiPixelVCalDB_h
#include <map>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelVCal.h"
//#include "CondTools/SiPixel/test/SiPixelVCalPixelId.h"

class SiPixelVCalDB : public edm::EDAnalyzer {
public:

  explicit SiPixelVCalDB(const edm::ParameterSet& conf);
  explicit SiPixelVCalDB();
  virtual ~SiPixelVCalDB();
  virtual void beginJob();
  virtual void endJob();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

private:

  std::string recordName_;
  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters BPixParameters_;
  Parameters FPixParameters_;

};

#endif
