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
  for(auto & thePixelGeomFactor : thePixelGeomFactors_) {
    string det = thePixelGeomFactor.getParameter<string>("det");
    thePixelGeomFactor.exists("layer") ? layer = thePixelGeomFactor.getParameter<unsigned int>("layer") : layer = LAYER;
    thePixelGeomFactor.exists("ladder") ? ladder = thePixelGeomFactor.getParameter<unsigned int>("ladder") : ladder = LADDER;
    thePixelGeomFactor.exists("module") ? module = thePixelGeomFactor.getParameter<unsigned int>("module") : module = MODULE;
    thePixelGeomFactor.exists("side") ? side = thePixelGeomFactor.getParameter<unsigned int>("side") : side = SIDE;
    thePixelGeomFactor.exists("disk") ? disk = thePixelGeomFactor.getParameter<unsigned int>("disk") : disk = DISK;
    thePixelGeomFactor.exists("blade") ? blade = thePixelGeomFactor.getParameter<unsigned int>("blade") : blade = BLADE;
    thePixelGeomFactor.exists("panel") ? panel = thePixelGeomFactor.getParameter<unsigned int>("panel") : panel = PANEL;
    double factor = thePixelGeomFactor.getParameter<double>("factor");
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
  for(auto & theColGeomFactor : theColGeomFactors_) {
    string det = theColGeomFactor.getParameter<string>("det");
    theColGeomFactor.exists("layer") ? layer = theColGeomFactor.getParameter<unsigned int>("layer") : layer = LAYER;
    theColGeomFactor.exists("ladder") ? ladder = theColGeomFactor.getParameter<unsigned int>("ladder") : ladder = LADDER;
    theColGeomFactor.exists("module") ? module = theColGeomFactor.getParameter<unsigned int>("module") : module = MODULE;
    theColGeomFactor.exists("side") ? side = theColGeomFactor.getParameter<unsigned int>("side") : side = SIDE;
    theColGeomFactor.exists("disk") ? disk = theColGeomFactor.getParameter<unsigned int>("disk") : disk = DISK;
    theColGeomFactor.exists("blade") ? blade = theColGeomFactor.getParameter<unsigned int>("blade") : blade = BLADE;
    theColGeomFactor.exists("panel") ? panel = theColGeomFactor.getParameter<unsigned int>("panel") : panel = PANEL;
    double factor = theColGeomFactor.getParameter<double>("factor");
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
  for(auto & theChipGeomFactor : theChipGeomFactors_) {
    string det = theChipGeomFactor.getParameter<string>("det");
    theChipGeomFactor.exists("layer") ? layer = theChipGeomFactor.getParameter<unsigned int>("layer") : layer = LAYER;
    theChipGeomFactor.exists("ladder") ? ladder = theChipGeomFactor.getParameter<unsigned int>("ladder") : ladder = LADDER;
    theChipGeomFactor.exists("module") ? module = theChipGeomFactor.getParameter<unsigned int>("module") : module = MODULE;
    theChipGeomFactor.exists("side") ? side = theChipGeomFactor.getParameter<unsigned int>("side") : side = SIDE;
    theChipGeomFactor.exists("disk") ? disk = theChipGeomFactor.getParameter<unsigned int>("disk") : disk = DISK;
    theChipGeomFactor.exists("blade") ? blade = theChipGeomFactor.getParameter<unsigned int>("blade") : blade = BLADE;
    theChipGeomFactor.exists("panel") ? panel = theChipGeomFactor.getParameter<unsigned int>("panel") : panel = PANEL;
    double factor = theChipGeomFactor.getParameter<double>("factor");
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
  for(auto & it : thePUEfficiency_) {
    string det = it.getParameter<string>("det");
    it.exists("layer") ? layer = it.getParameter<unsigned int>("layer") : layer = LAYER;
    it.exists("ladder") ? ladder = it.getParameter<unsigned int>("ladder") : ladder = LADDER;
    it.exists("module") ? module = it.getParameter<unsigned int>("module") : module = MODULE;
    it.exists("side") ? side = it.getParameter<unsigned int>("side") : side = SIDE;
    it.exists("disk") ? disk = it.getParameter<unsigned int>("disk") : disk = DISK;
    it.exists("blade") ? blade = it.getParameter<unsigned int>("blade") : blade = BLADE;
    it.exists("panel") ? panel = it.getParameter<unsigned int>("panel") : panel = PANEL;
    std::vector<double> factor = it.getParameter<std::vector<double> >("factor");
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
