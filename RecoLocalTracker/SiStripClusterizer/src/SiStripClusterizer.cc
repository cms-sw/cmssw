// File: SiStripClusterizer.cc
// Description:  see SiStripClusterizer.h
// Author:  O. Gutsche
// Creation Date:  OGU Aug. 1 2005 Initial version.
//
//--------------------------------------------

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizer.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

namespace cms
{
  SiStripClusterizer::SiStripClusterizer(edm::ParameterSet const& conf) : 
    conf_(conf),
    SiStripClusterizerAlgorithm_(conf) {

    edm::LogInfo("SiStripClusterizer") << "[SiStripClusterizer::SiStripClusterizer] Constructing object...";
    
    produces< edm::DetSetVector<SiStripCluster> > ();
  }

  // Virtual destructor needed.
  SiStripClusterizer::~SiStripClusterizer() { 
    edm::LogInfo("SiStripClusterizer") << "[SiStripClusterizer::~SiStripClusterizer] Destructing object...";
  }  

  // Functions that gets called by framework every event
  void SiStripClusterizer::produce(edm::Event& e, const edm::EventSetup& es)
  {
    // Step A: Get ESObject 
    edm::ESHandle<SiStripGain> gainHandle;
    edm::ESHandle<SiStripNoises> noiseHandle;
    edm::ESHandle<SiStripQuality> qualityHandle;


    es.get<SiStripGainRcd>().get(gainHandle);
    es.get<SiStripNoisesRcd>().get(noiseHandle);
    if (conf_.getParameter<std::string>("SiStripQualityLabel")!="NULL")
      es.get<SiStripQualityRcd>().get(conf_.getParameter<std::string>("SiStripQualityLabel"),qualityHandle);
    else{
      edm::ESHandle<SiStripQuality> tmpqualityHandle(&emptyQuality);
      tmpqualityHandle.swap(qualityHandle);
    }

    // Step B: Get Inputs 
    edm::Handle< edm::DetSetVector<SiStripDigi> >  input;

    // Step C: produce output product
    std::vector< edm::DetSet<SiStripCluster> > vSiStripCluster;
    vSiStripCluster.reserve(10000);
    typedef std::vector<edm::ParameterSet> Parameters;
    Parameters DigiProducersList = conf_.getParameter<Parameters>("DigiProducersList");
    Parameters::iterator itDigiProducersList = DigiProducersList.begin();
    for(; itDigiProducersList != DigiProducersList.end(); ++itDigiProducersList ) {
      std::string digiProducer = itDigiProducersList->getParameter<std::string>("DigiProducer");
      std::string digiLabel = itDigiProducersList->getParameter<std::string>("DigiLabel");
      e.getByLabel(digiProducer,digiLabel,input);  //FIXME: fix this label	
      if (input->size())
	SiStripClusterizerAlgorithm_.run(*input,vSiStripCluster, noiseHandle, gainHandle, qualityHandle);
    }
    
    // Step D: create and fill output collection
    std::auto_ptr< edm::DetSetVector<SiStripCluster> > output(new edm::DetSetVector<SiStripCluster>(vSiStripCluster) );

    // Step D: write output to file
    e.put(output);
  }
}
