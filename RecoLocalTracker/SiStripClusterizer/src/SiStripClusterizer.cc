// File: SiStripClusterizer.cc
// Description:  see SiStripClusterizer.h
// Author:  O. Gutsche
// Creation Date:  OGU Aug. 1 2005 Initial version.
//
//--------------------------------------------

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizer.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h" //@@ To assure backward compatibility

namespace cms
{
  SiStripClusterizer::SiStripClusterizer(edm::ParameterSet const& conf) : 
    conf_(conf),
    SiStripClusterizerAlgorithm_(conf) ,
    SiStripNoiseService_(conf){

    edm::LogInfo("SiStripClusterizer") << "[SiStripClusterizer::SiStripClusterizer] Constructing object...";

    produces< edm::DetSetVector<SiStripCluster> > ();
    produces<SiStripClusterCollection>(); //@@ To assure backward compatibility
  }

  // Virtual destructor needed.
  SiStripClusterizer::~SiStripClusterizer() { 
    edm::LogInfo("SiStripClusterizer") << "[SiStripClusterizer::~SiStripClusterizer] Destructing object...";
  }  

  //Get at the beginning
  void SiStripClusterizer::beginJob( const edm::EventSetup& es ) {
    edm::LogInfo("SiStripClusterizer") << "[SiStripClusterizer::beginJob]";
    
    SiStripNoiseService_.configure(es);
    SiStripClusterizerAlgorithm_.configure(&SiStripNoiseService_);
  }

  // Functions that gets called by framework every event
  void SiStripClusterizer::produce(edm::Event& e, const edm::EventSetup& es)
  {
    // retrieve producer name of input StripDigiCollection
    std::string digiProducer = conf_.getParameter<std::string>("DigiProducer");

    // Step A: create empty output collection
    std::auto_ptr< edm::DetSetVector<SiStripCluster> > output(new edm::DetSetVector<SiStripCluster> );
    std::auto_ptr<SiStripClusterCollection> output_old(new SiStripClusterCollection);//@@ To assure backward compatibility



    // Step B: Get Inputs 
    edm::Handle< edm::DetSetVector<SiStripDigi> >    zsDigis;
    edm::Handle< edm::DetSetVector<SiStripDigi> >    vrDigis;
    edm::Handle< edm::DetSetVector<SiStripDigi> >    prDigis;
    edm::Handle< edm::DetSetVector<SiStripDigi> >    smDigis;

    if (digiProducer=="stripdigi"){
      e.getByLabel(digiProducer,"stripdigi",zsDigis);  //FIXME: fix this label
    }
    else{
      digiProducer="RawToDigi"; //FIXME: fix this label
      e.getByLabel(digiProducer,"ZeroSuppressed",zsDigis);  //FIXME: fix this label
      digiProducer="theSiStripZeroSuppression"; //FIXME: fix this label
      e.getByLabel(digiProducer,"fromVirginRaw"   ,vrDigis);
      e.getByLabel(digiProducer,"fromProcessedRaw",prDigis);
      e.getByLabel(digiProducer,"fromScopeMode"   ,smDigis);
    }
    
    // Step C: Get ESObject 
    SiStripNoiseService_.setESObjects(es);

    // Step C: Invoke the strip clusterizer algorithm
    if (zsDigis->size())
      SiStripClusterizerAlgorithm_.run(*zsDigis,*output);
    if (vrDigis->size())
      SiStripClusterizerAlgorithm_.run(*vrDigis,*output);
    if (prDigis->size())
      SiStripClusterizerAlgorithm_.run(*prDigis,*output);
    if (smDigis->size())
      SiStripClusterizerAlgorithm_.run(*smDigis,*output);
    
    // Step D: write output to file
    if ( output->size() )
      {
	//@@ To assure backward compatibility
	for (edm::DetSetVector<SiStripCluster>::const_iterator iter=output->begin();iter!=output->end();iter++)
	  {
	    std::vector<SiStripCluster> collector;
	    for (edm::DetSet<SiStripCluster>::const_iterator jter=iter->data.begin();jter!=iter->data.end();jter++)
	      collector.push_back(*jter);
	    SiStripClusterCollection::Range inputRange;
	    inputRange.first = collector.begin();
	    inputRange.second = collector.end();
	    output_old->put(inputRange,iter->id);
	  }
	e.put(output_old);
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	e.put(output);
      }
  }

}
