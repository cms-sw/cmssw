#include "RecoLocalTracker/SiStripClusterizer/plugins/SiStripClusterizer.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithmFactory.h"

SiStripClusterizer::
SiStripClusterizer(const edm::ParameterSet& conf) 
  : inputTags( conf.getParameter<std::vector<edm::InputTag> >("DigiProducersList") ),
    algorithm( StripClusterizerAlgorithmFactory::create(conf) ) {
  produces< edmNew::DetSetVector<SiStripCluster> > ();
}

void SiStripClusterizer::
produce(edm::Event& event, const edm::EventSetup& es)  {

  std::auto_ptr< edmNew::DetSetVector<SiStripCluster> > output(new edmNew::DetSetVector<SiStripCluster>());
  output->reserve(10000,4*10000);

  edm::Handle< edm::DetSetVector<SiStripDigi> >     inputOld;  
  edm::Handle< edmNew::DetSetVector<SiStripDigi> >  inputNew;  

  algorithm->initialize(es);  
  if( findInput(inputOld, event) ) algorithm->clusterize(*inputOld, *output); else 
    if( findInput(inputNew, event) ) algorithm->clusterize(*inputNew, *output); else
      edm::LogWarning("Input Not Found");

  edm::LogInfo("Output") << output->dataSize() << " clusters from " 
			 << output->size()     << " modules";
  event.put(output);
}

template<class T>
inline
bool SiStripClusterizer::
findInput(edm::Handle<T>& handle, const edm::Event& e) {

  for(std::vector<edm::InputTag>::const_iterator 
	inputTag = inputTags.begin();  inputTag != inputTags.end();  inputTag++) {

    e.getByLabel(*inputTag, handle);
    if( handle.isValid() && !handle->empty() ) {
      edm::LogInfo("Input") << *inputTag;
      return true;
    }
  }
  return false;
}
