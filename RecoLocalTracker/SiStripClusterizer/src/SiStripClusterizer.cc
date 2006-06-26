// File: SiStripClusterizer.cc
// Description:  see SiStripClusterizer.h
// Author:  O. Gutsche
// Creation Date:  OGU Aug. 1 2005 Initial version.
//
//--------------------------------------------

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizer.h"

namespace cms
{
  SiStripClusterizer::SiStripClusterizer(edm::ParameterSet const& conf) : 
    conf_(conf),
    SiStripClusterizerAlgorithm_(conf) ,
    SiStripNoiseService_(conf){

    edm::LogInfo("SiStripClusterizer") << "[SiStripClusterizer::SiStripClusterizer] Constructing object...";

    produces< edm::DetSetVector<SiStripCluster> > ();
  }

  // Virtual destructor needed.
  SiStripClusterizer::~SiStripClusterizer() { 
    edm::LogInfo("SiStripClusterizer") << "[SiStripClusterizer::~SiStripClusterizer] Destructing object...";
  }  

  //Get at the beginning
  void SiStripClusterizer::beginJob( const edm::EventSetup& es ) {
    edm::LogInfo("SiStripClusterizer") << "[SiStripClusterizer::beginJob]";
    
    //SiStripNoiseService_.configure(es); @not needed anymore, REMOVE
    SiStripClusterizerAlgorithm_.configure(&SiStripNoiseService_);
  }

  // Functions that gets called by framework every event
  void SiStripClusterizer::produce(edm::Event& e, const edm::EventSetup& es)
  {
    // retrieve producer name of input StripDigiCollection
    //    std::string digiProducer = conf_.getParameter<std::string>("DigiProducer");

    // Step A: create empty output collection
    std::auto_ptr< edm::DetSetVector<SiStripCluster> > output(new edm::DetSetVector<SiStripCluster> );

    // Step B: Get ESObject 
    SiStripNoiseService_.setESObjects(es);

    // Step C: Get Inputs 
    edm::Handle< edm::DetSetVector<SiStripDigi> >  input;

    typedef std::vector<edm::ParameterSet> Parameters;
    Parameters DigiProducersList = conf_.getParameter<Parameters>("DigiProducersList");
    Parameters::iterator itDigiProducersList = DigiProducersList.begin();
    for(; itDigiProducersList != DigiProducersList.end(); ++itDigiProducersList ) {
      std::string digiProducer = itDigiProducersList->getParameter<std::string>("DigiProducer");
      std::string digiLabel = itDigiProducersList->getParameter<std::string>("DigiLabel");
      e.getByLabel(digiProducer,digiLabel,input);  //FIXME: fix this label	
      if (input->size())
	SiStripClusterizerAlgorithm_.run(*input,*output);
    }
    

    // Step D: write output to file
    e.put(output);
  }
}
