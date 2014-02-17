// -*- C++ -*-
// Package:    SiPixelESProducers
// Class:      SiPixelDetInfoFileWriter
// Original Author:  V.Chiochia (adapted from the Strip version by G.Bruno)
//         Created:  Mon May 20 10:04:31 CET 2007
// $Id: SiPixelDetInfoFileWriter.cc,v 1.4 2010/01/14 09:36:57 ursl Exp $

#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileWriter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h" 
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"


using namespace cms;
using namespace std;


SiPixelDetInfoFileWriter::SiPixelDetInfoFileWriter(const edm::ParameterSet& iConfig) {

  
  edm::LogInfo("SiPixelDetInfoFileWriter::SiPixelDetInfoFileWriter");

  filePath_ = iConfig.getUntrackedParameter<std::string>("FilePath",std::string("SiPixelDetInfo.dat"));

}


SiPixelDetInfoFileWriter::~SiPixelDetInfoFileWriter(){

   edm::LogInfo("SiPixelDetInfoFileWriter::~SiPixelDetInfoFileWriter");
}



void SiPixelDetInfoFileWriter::beginRun(const edm::Run &run , const edm::EventSetup &iSetup){

  outputFile_.open(filePath_.c_str());

  if (outputFile_.is_open()){

    edm::ESHandle<TrackerGeometry> pDD;

    iSetup.get<TrackerDigiGeometryRecord>().get( pDD );

    edm::LogInfo("SiPixelDetInfoFileWriter::beginJob - got geometry  ")<<std::endl;    
    edm::LogInfo("SiPixelDetInfoFileWriter") <<" There are "<<pDD->detUnits().size() <<" detectors"<<std::endl;
    
    int nPixelDets = 0;

    for(TrackerGeometry::DetUnitContainer::const_iterator it = pDD->detUnits().begin(); it != pDD->detUnits().end(); it++){
  
      const PixelGeomDetUnit* mit = dynamic_cast<PixelGeomDetUnit*>(*it);

      if(mit!=0){
	nPixelDets++;
      const PixelTopology & topol = mit->specificTopology();       
      // Get the module sizes.
      int nrows = topol.nrows();      // rows in x
      int ncols = topol.ncolumns();   // cols in y      
      uint32_t detid=(mit->geographicalId()).rawId();
      
      
      outputFile_ << detid << " "<< ncols << " " << nrows << "\n";
      
      }
    }    
    outputFile_.close();
    edm::LogInfo("SiPixelDetInfoFileWriter::beginJob - Loop finished. ")<< nPixelDets << " Pixel DetUnits found " << std::endl;
  }
  
  else {

    edm::LogError("SiPixelDetInfoFileWriter::beginJob - Unable to open file")<<endl;
    return;
  
  }

}


void SiPixelDetInfoFileWriter::beginJob() {

}

void SiPixelDetInfoFileWriter::analyze(const edm::Event &, const edm::EventSetup &) {

}
