// -*- C++ -*-
// Package:    SiStripCommon
// Class:      SiStripDetInfoFileWriter
// Original Author:  G. Bruno
//         Created:  Mon May 20 10:04:31 CET 2007

#include "CalibTracker/SiStripCommon/plugins/SiStripDetInfoFileWriter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h" 
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"


using namespace cms;
using namespace std;


SiStripDetInfoFileWriter::SiStripDetInfoFileWriter(const edm::ParameterSet& iConfig) {

  
  edm::LogInfo("SiStripDetInfoFileWriter::SiStripDetInfoFileWriter");

  filePath_ = iConfig.getUntrackedParameter<std::string>("FilePath",std::string("SiStripDetInfo.dat"));

}


SiStripDetInfoFileWriter::~SiStripDetInfoFileWriter(){

   edm::LogInfo("SiStripDetInfoFileWriter::~SiStripDetInfoFileWriter");
}



void SiStripDetInfoFileWriter::beginRun(const edm::Run&, const edm::EventSetup& iSetup){


  outputFile_.open(filePath_.c_str());

  if (outputFile_.is_open()){


    edm::ESHandle<TrackerGeometry> pDD;

    iSetup.get<TrackerDigiGeometryRecord>().get( pDD );

    edm::LogInfo("SiStripDetInfoFileWriter::beginRun - got geometry  ")<<std::endl;

    
    edm::LogInfo("SiStripDetInfoFileWriter") <<" There are "<<pDD->detUnits().size() <<" detectors"<<std::endl;
    
    for(TrackerGeometry::DetUnitContainer::const_iterator it = pDD->detUnits().begin(); it != pDD->detUnits().end(); it++){
  
      const StripGeomDetUnit* mit = dynamic_cast<StripGeomDetUnit const *>(*it);

      if(mit!=0){

	uint32_t detid=(mit->geographicalId()).rawId();
	double stripLength = mit->specificTopology().stripLength();
	unsigned short numberOfAPVs= mit->specificTopology().nstrips()/128;
	float thickness=mit->specificSurface().bounds().thickness();


	if(numberOfAPVs<1 || numberOfAPVs>6 ) {
	  edm::LogError("SiStripDetInfoFileWriter")<<" Problem with Number of strips in detector.. "<< mit->specificTopology().nstrips() <<  "Will not write this entry to file"<< endl;
	  continue;
	}

	outputFile_ << detid << " "<< numberOfAPVs << " " << stripLength << " "<< thickness << "\n";

      }

    }
    
    outputFile_.close();

  }
  
  else {

    edm::LogError("SiStripDetInfoFileWriter::beginRun - Unable to open file")<<endl;
    return;
  
  }

}


