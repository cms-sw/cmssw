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

#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


//Added by D. Giordano
//FIXME: the first 2 include are needed??
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"

#include <iostream> 

namespace cms
{

  SiStripClusterizer::SiStripClusterizer(edm::ParameterSet const& conf) : 
    siStripClusterizerAlgorithm_(conf) ,
    conf_(conf)
  {
    produces<SiStripClusterCollection>();
  }


  // Virtual destructor needed.
  SiStripClusterizer::~SiStripClusterizer() { }  

  //Get at the beginning Calibration data (pedestals)
  void SiStripClusterizer::beginJob( const edm::EventSetup& iSetup ) {

    std::cout << "BeginJob method " << std::endl;
    std::cout << "Here I am " << std::endl;
  
    //edm::ESHandle<SiStripPedestals> ped;
    iSetup.get<SiStripPedestalsRcd>().get(ped);
    
    SiStripPedestalsMapIterator mapit = (*ped).m_pedestals.begin();
    for (;mapit!=(*ped).m_pedestals.end();mapit++)
      {
	unsigned int detid = (*mapit).first;
	std::cout << "detid " <<  detid << " # Strip " << (*mapit).second.size()<<std::endl;
	//SiStripPedestalsVector theSiStripVector =  (*mapit).second;     
	const SiStripPedestalsVector theSiStripVector =  (*ped).getSiStripPedestalsVector(detid);
	
	int strip=0;
	SiStripPedestalsVectorIterator iter=theSiStripVector.begin();
	//for(; iter!=theSiStripVector.end(); iter++)
	  {
	    std::cout << " strip " << strip++ << " =\t"
		      << iter->getPed()       << " \t" 
		      << iter->getNoise()     << " \t" 
		      << iter->getLowTh()     << " \t" 
		      << iter->getHighTh()    << " \t" 
		      << iter->getDisable()   << " \t" 
		      << std::endl; 	    
	  } 
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
    siStripClusterizerAlgorithm_.run(stripDigis.product(),*output,ped);

    // Step D: write output to file
    e.put(output);

  }

}
