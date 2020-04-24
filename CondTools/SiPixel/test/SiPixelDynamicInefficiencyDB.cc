#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <limits>
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondTools/SiPixel/test/SiPixelDynamicInefficiencyDB.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelDynamicInefficiency.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

using namespace std;
using namespace edm;

  //Constructor

SiPixelDynamicInefficiencyDB::SiPixelDynamicInefficiencyDB(edm::ParameterSet const& conf) : 
  conf_(conf){
	recordName_ = conf_.getUntrackedParameter<std::string>("record","SiPixelDynamicInefficiencyRcd");

       thePixelGeomFactors_ = conf_.getUntrackedParameter<Parameters>("thePixelGeomFactors");
       theColGeomFactors_ = conf_.getUntrackedParameter<Parameters>("theColGeomFactors");
       theChipGeomFactors_ = conf_.getUntrackedParameter<Parameters>("theChipGeomFactors");
       thePUEfficiency_ = conf_.getUntrackedParameter<Parameters>("thePUEfficiency");
       theInstLumiScaleFactor_ = conf_.getUntrackedParameter<double>("theInstLumiScaleFactor");
}

  //BeginJob

void SiPixelDynamicInefficiencyDB::beginJob(){
  
}

// Virtual destructor needed.

SiPixelDynamicInefficiencyDB::~SiPixelDynamicInefficiencyDB() {  

}  

// Analyzer: Functions that gets called by framework every event

void SiPixelDynamicInefficiencyDB::analyze(const edm::Event& e, const edm::EventSetup& es)
{

	SiPixelDynamicInefficiency* DynamicInefficiency = new SiPixelDynamicInefficiency();


  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  uint32_t max = numeric_limits<uint32_t>::max();
  uint32_t mask;
  uint32_t layer, LAYER = 0;
  uint32_t ladder, LADDER = 0;
  uint32_t module, MODULE = 0;
  uint32_t side, SIDE = 0;
  uint32_t disk, DISK = 0;
  uint32_t blade, BLADE = 0;
  uint32_t panel, PANEL = 0;

  //Put BPix masks
  mask = tTopo->pxbDetId(max,LADDER,MODULE).rawId();
  DynamicInefficiency->putDetIdmask(mask);
  mask = tTopo->pxbDetId(LAYER,max,MODULE).rawId();
  DynamicInefficiency->putDetIdmask(mask);
  mask = tTopo->pxbDetId(LAYER,LADDER,max).rawId();
  DynamicInefficiency->putDetIdmask(mask);
  //Put FPix masks
  mask = tTopo->pxfDetId(max,DISK,BLADE,PANEL,MODULE).rawId();
  DynamicInefficiency->putDetIdmask(mask);
  mask = tTopo->pxfDetId(SIDE,max,BLADE,PANEL,MODULE).rawId();
  DynamicInefficiency->putDetIdmask(mask);
  mask = tTopo->pxfDetId(SIDE,DISK,max,PANEL,MODULE).rawId();
  DynamicInefficiency->putDetIdmask(mask);
  mask = tTopo->pxfDetId(SIDE,DISK,BLADE,max,MODULE).rawId();
  DynamicInefficiency->putDetIdmask(mask);
  mask = tTopo->pxfDetId(SIDE,DISK,BLADE,PANEL,max).rawId();
  DynamicInefficiency->putDetIdmask(mask);
  
  //Put PixelGeomFactors
  for(Parameters::iterator it = thePixelGeomFactors_.begin(); it != thePixelGeomFactors_.end(); ++it) {
    string det = it->getParameter<string>("det");
    it->exists("layer") ? layer = it->getParameter<unsigned int>("layer") : layer = LAYER;
    it->exists("ladder") ? ladder = it->getParameter<unsigned int>("ladder") : ladder = LADDER;
    it->exists("module") ? module = it->getParameter<unsigned int>("module") : module = MODULE;
    it->exists("side") ? side = it->getParameter<unsigned int>("side") : side = SIDE;
    it->exists("disk") ? disk = it->getParameter<unsigned int>("disk") : disk = DISK;
    it->exists("blade") ? blade = it->getParameter<unsigned int>("blade") : blade = BLADE;
    it->exists("panel") ? panel = it->getParameter<unsigned int>("panel") : panel = PANEL;
    double factor = it->getParameter<double>("factor");
    if (det == "bpix") {
      DetId detID=tTopo->pxbDetId(layer,ladder,module);
      std::cout<<"Putting Pixel geom BPix layer "<<layer<<" ladder "<<ladder<<" module "<<module<<" factor "<<factor<<std::endl;
      DynamicInefficiency->putPixelGeomFactor(detID.rawId(),factor);
    }
    else if (det == "fpix") {
      DetId detID=tTopo->pxfDetId(side, disk, blade, panel, module);
      std::cout<<"Putting Pixel geom FPix side "<<side<<" disk "<<disk<<" blade "<<blade<<" panel "<<panel<<" module "<<module<<" factor "<<factor<<std::endl;
      DynamicInefficiency->putPixelGeomFactor(detID.rawId(),factor);
    }
    else edm::LogError("SiPixelDynamicInefficiencyDB")<<"SiPixelDynamicInefficiencyDB input detector part is neither bpix nor fpix"<<std::endl;
  }

  //Put ColumnGeomFactors
  for(Parameters::iterator it = theColGeomFactors_.begin(); it != theColGeomFactors_.end(); ++it) {
    string det = it->getParameter<string>("det");
    it->exists("layer") ? layer = it->getParameter<unsigned int>("layer") : layer = LAYER;
    it->exists("ladder") ? ladder = it->getParameter<unsigned int>("ladder") : ladder = LADDER;
    it->exists("module") ? module = it->getParameter<unsigned int>("module") : module = MODULE;
    it->exists("side") ? side = it->getParameter<unsigned int>("side") : side = SIDE;
    it->exists("disk") ? disk = it->getParameter<unsigned int>("disk") : disk = DISK;
    it->exists("blade") ? blade = it->getParameter<unsigned int>("blade") : blade = BLADE;
    it->exists("panel") ? panel = it->getParameter<unsigned int>("panel") : panel = PANEL;
    double factor = it->getParameter<double>("factor");
    if (det == "bpix") {
      DetId detID=tTopo->pxbDetId(layer,ladder,module);
      std::cout<<"Putting Column geom BPix layer "<<layer<<" ladder "<<ladder<<" module "<<module<<" factor "<<factor<<std::endl;
      DynamicInefficiency->putColGeomFactor(detID.rawId(),factor);
    }
    else if (det == "fpix") {
      DetId detID=tTopo->pxfDetId(side, disk, blade, panel, module);
      std::cout<<"Putting Column geom FPix side "<<side<<" disk "<<disk<<" blade "<<blade<<" panel "<<panel<<" module "<<module<<" factor "<<factor<<std::endl;
      DynamicInefficiency->putColGeomFactor(detID.rawId(),factor);
    }
    else edm::LogError("SiPixelDynamicInefficiencyDB")<<"SiPixelDynamicInefficiencyDB input detector part is neither bpix nor fpix"<<std::endl;
  }

  //Put ChipGeomFactors
  for(Parameters::iterator it = theChipGeomFactors_.begin(); it != theChipGeomFactors_.end(); ++it) {
    string det = it->getParameter<string>("det");
    it->exists("layer") ? layer = it->getParameter<unsigned int>("layer") : layer = LAYER;
    it->exists("ladder") ? ladder = it->getParameter<unsigned int>("ladder") : ladder = LADDER;
    it->exists("module") ? module = it->getParameter<unsigned int>("module") : module = MODULE;
    it->exists("side") ? side = it->getParameter<unsigned int>("side") : side = SIDE;
    it->exists("disk") ? disk = it->getParameter<unsigned int>("disk") : disk = DISK;
    it->exists("blade") ? blade = it->getParameter<unsigned int>("blade") : blade = BLADE;
    it->exists("panel") ? panel = it->getParameter<unsigned int>("panel") : panel = PANEL;
    double factor = it->getParameter<double>("factor");
    if (det == "bpix") {
      DetId detID=tTopo->pxbDetId(layer,ladder,module);
      std::cout<<"Putting Chip geom BPix layer "<<layer<<" ladder "<<ladder<<" module "<<module<<" factor "<<factor<<std::endl;
      DynamicInefficiency->putChipGeomFactor(detID.rawId(),factor);
    }
    else if (det == "fpix") {
      DetId detID=tTopo->pxfDetId(side, disk, blade, panel, module);
      std::cout<<"Putting Chip geom FPix side "<<side<<" disk "<<disk<<" blade "<<blade<<" panel "<<panel<<" module "<<module<<" factor "<<factor<<std::endl;
      DynamicInefficiency->putChipGeomFactor(detID.rawId(),factor);
    }
    else edm::LogError("SiPixelDynamicInefficiencyDB")<<"SiPixelDynamicInefficiencyDB input detector part is neither bpix nor fpix"<<std::endl;
  }

  //Put PUFactors
  for(Parameters::iterator it = thePUEfficiency_.begin(); it != thePUEfficiency_.end(); ++it) {
    string det = it->getParameter<string>("det");
    it->exists("layer") ? layer = it->getParameter<unsigned int>("layer") : layer = LAYER;
    it->exists("ladder") ? ladder = it->getParameter<unsigned int>("ladder") : ladder = LADDER;
    it->exists("module") ? module = it->getParameter<unsigned int>("module") : module = MODULE;
    it->exists("side") ? side = it->getParameter<unsigned int>("side") : side = SIDE;
    it->exists("disk") ? disk = it->getParameter<unsigned int>("disk") : disk = DISK;
    it->exists("blade") ? blade = it->getParameter<unsigned int>("blade") : blade = BLADE;
    it->exists("panel") ? panel = it->getParameter<unsigned int>("panel") : panel = PANEL;
    std::vector<double> factor = it->getParameter<std::vector<double> >("factor");
    if (det == "bpix") {
      DetId detID=tTopo->pxbDetId(layer,ladder,module);
      std::cout<<"Putting PU efficiency BPix layer "<<layer<<" ladder "<<ladder<<" module "<<module<<" factor size "<<factor.size()<<std::endl;
      DynamicInefficiency->putPUFactor(detID.rawId(),factor);
    }
    else if (det == "fpix") {
      DetId detID=tTopo->pxfDetId(side, disk, blade, panel, module);
      std::cout<<"Putting PU efficiency FPix side "<<side<<" disk "<<disk<<" blade "<<blade<<" panel "<<panel<<" module "<<module<<" factor size "<<factor.size()<<std::endl;
      DynamicInefficiency->putPUFactor(detID.rawId(),factor);
    }
  }
  //Put theInstLumiScaleFactor
  DynamicInefficiency->puttheInstLumiScaleFactor(theInstLumiScaleFactor_);

	edm::Service<cond::service::PoolDBOutputService> mydbservice;
	if( mydbservice.isAvailable() ){
		try{
			if( mydbservice->isNewTagRequest(recordName_) ){
				mydbservice->createNewIOV<SiPixelDynamicInefficiency>(DynamicInefficiency,
									       mydbservice->beginOfTime(),
									       mydbservice->endOfTime(),
									       recordName_);
			} else {
				mydbservice->appendSinceTime<SiPixelDynamicInefficiency>(DynamicInefficiency,
										  mydbservice->currentTime(),
										  recordName_);
			}
		}catch(const cond::Exception& er){
			edm::LogError("SiPixelDynamicInefficiencyDB")<<er.what()<<std::endl;
		}catch(const std::exception& er){
			edm::LogError("SiPixelDynamicInefficiencyDB")<<"caught std::exception "<<er.what()<<std::endl;
		}catch(...){
			edm::LogError("SiPixelDynamicInefficiencyDB")<<"Funny error"<<std::endl;
		}
	}else{
		edm::LogError("SiPixelDynamicInefficiencyDB")<<"Service is unavailable"<<std::endl;
	}
   
}
void SiPixelDynamicInefficiencyDB::endJob(){
}
