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
    SiStripClusterizerAlgorithm_(conf) {

    edm::LogInfo("SiStripClusterizer") << "[SiStripClusterizer::SiStripClusterizer] Constructing object...";
    
    produces< edmNew::DetSetVector<SiStripCluster> > ();
  }

  // Virtual destructor needed.
  SiStripClusterizer::~SiStripClusterizer() { 
    edm::LogInfo("SiStripClusterizer") << "[SiStripClusterizer::~SiStripClusterizer] Destructing object...";
  }  

  // Functions that gets called by framework every event
  void SiStripClusterizer::produce(edm::Event& e, const edm::EventSetup& es)
  {

    // Step B: Get Inputs 
    edm::Handle< edm::DetSetVector<SiStripDigi> >  input;

    // Step C: produce output product
    std::auto_ptr< edmNew::DetSetVector<SiStripCluster> > output(new edmNew::DetSetVector<SiStripCluster>());
    output->reserve(10000,4*10000); //FIXME

    typedef std::vector<edm::ParameterSet> Parameters;
    Parameters DigiProducersList = conf_.getParameter<Parameters>("DigiProducersList");
    Parameters::iterator itDigiProducersList = DigiProducersList.begin();
    for(; itDigiProducersList != DigiProducersList.end(); ++itDigiProducersList ) {
      std::string digiProducer = itDigiProducersList->getParameter<std::string>("DigiProducer");
      std::string digiLabel = itDigiProducersList->getParameter<std::string>("DigiLabel");
      e.getByLabel(digiProducer,digiLabel,input);  //FIXME: fix this label	
      if (input->size())
	SiStripClusterizerAlgorithm_.run(*input,*output, es);
    }
    
    // Step D: write output to file
    e.put(output);
  }
}
