// File: SiStripClusterizer.cc
// Description:  see SiStripClusterizer.h
// Author:  O. Gutsche
// Creation Date:  OGU Aug. 1 2005 Initial version.
//
//--------------------------------------------
#include <memory>
#include <string>

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizer.h"
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "FWCore/Framework/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


//Added by D. Giordano
//FIXME: the first 2 include are needed??
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"

#include <iostream> 

namespace cms
{
  SiStripClusterizer::SiStripClusterizer(edm::ParameterSet const& conf) : 
    siStripClusterizerAlgorithm_(conf) ,
    conf_(conf){
    produces<SiStripClusterCollection>();
  }

  // Virtual destructor needed.
  SiStripClusterizer::~SiStripClusterizer() { }  

  //Get at the beginning
  void SiStripClusterizer::beginJob( const edm::EventSetup& iSetup ) {
    std::cout << "BeginJob method " << std::endl;

    //Getting Geometry
    iSetup.get<TrackerDigiGeometryRecord>().get( pDD );
    cout <<" There are "<<pDD->dets().size() <<" detectors"<<endl;

    //Getting Calibration data (Noises and BadStrips Flag)
    bool UseNoiseBadStripFlagFromDB_=conf_.getParameter<bool>("UseNoiseBadStripFlagFromDB_");  
    if (UseNoiseBadStripFlagFromDB_==true){
      iSetup.get<SiStripNoisesRcd>().get(noise);
      //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
      // FIXME
      // Debug: show noise for DetIDs
      SiStripNoiseMapIterator mapit = noise->m_noises.begin();
      for (;mapit!=noise->m_noises.end();mapit++)
	{
	  unsigned int detid = (*mapit).first;
	  std::cout << "detid " <<  detid << " # Strip " << (*mapit).second.size()<<std::endl;
	  //SiStripNoiseVector theSiStripVector =  (*mapit).second;     
	  const SiStripNoiseVector theSiStripVector =  noise->getSiStripNoisesVector(detid);
	  
	  
	  int strip=0;
	  SiStripNoiseVectorIterator iter=theSiStripVector.begin();
	  //for(; iter!=theSiStripVector.end(); iter++)
	  {
	    std::cout << " strip " << strip++ << " =\t"
		      << iter->getNoise()     << " \t" 
		      << iter->getDisable()   << " \t" 
		      << std::endl; 	    
	  } 
	}
      //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    }
  }

  // Functions that gets called by framework every event
  void SiStripClusterizer::produce(edm::Event& e, const edm::EventSetup& es)
  {
    // retrieve producer name of input StripDigiCollection
    std::string digiProducer = conf_.getParameter<std::string>("DigiProducer");

    // Step A: Get Inputs 
    edm::Handle<StripDigiCollection> stripDigis;
    e.getByLabel(digiProducer, stripDigis);

    // Step B: create empty output collection
    std::auto_ptr<SiStripClusterCollection> output(new SiStripClusterCollection);

    // Step C: Invoke the strip clusterizer algorithm
    siStripClusterizerAlgorithm_.run(stripDigis.product(),*output,noise,pDD);

    // Step D: write output to file
    e.put(output);

  }

}
