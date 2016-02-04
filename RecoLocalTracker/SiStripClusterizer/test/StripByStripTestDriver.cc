
#include "RecoLocalTracker/SiStripClusterizer/test/StripByStripTestDriver.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/Framework/interface/Event.h"

StripByStripTestDriver::
StripByStripTestDriver(const edm::ParameterSet&conf)
  : inputTag( conf.getParameter<edm::InputTag>("DigiProducer") ),
    hlt( conf.getParameter<bool>("HLT") )//,
  /*hltFactory(0)*/ {

  if(hlt);// hltFactory = new SiStripClusterizerFactory(conf);
  else    algorithm  = StripClusterizerAlgorithmFactory::create(conf);
  
  produces<output_t>("");
}

StripByStripTestDriver::
~StripByStripTestDriver() {
  //if(hltFactory) delete hltFactory;
}

void StripByStripTestDriver::
produce(edm::Event& event, const edm::EventSetup& es) {

  std::auto_ptr<output_t> output(new output_t());
  output->reserve(10000,4*10000);

  edm::Handle< edm::DetSetVector<SiStripDigi> >  input;  
  event.getByLabel(inputTag, input);

  if(hlt);// hltFactory->eventSetup(es);  
  else    algorithm->initialize(es);

  for(edm::DetSetVector<SiStripDigi>::const_iterator 
	inputDetSet = input->begin(); inputDetSet != input->end(); inputDetSet++) {

    std::vector<SiStripCluster> clusters;
    if( hlt || algorithm->stripByStripBegin( inputDetSet->detId() ) ) {
      for( edm::DetSet<SiStripDigi>::const_iterator
	     digi = inputDetSet->begin(); digi != inputDetSet->end(); digi++ ) {
	if(hlt);// hltFactory->algorithm()->add(clusters, inputDetSet->detId(), digi->strip(), digi->adc());
	else    algorithm->stripByStripAdd(digi->strip(), digi->adc(), clusters);
      }
      if(hlt);// hltFactory->algorithm()->endDet(clusters, inputDetSet->detId());
      else    algorithm->stripByStripEnd(clusters);
    }

    if(!clusters.empty()) {
      output_t::FastFiller filler(*output, inputDetSet->detId());
      for( unsigned i=0; i<clusters.size(); i++) filler.push_back(clusters[i]);
    }
  }


  edm::LogInfo("Output") << output->dataSize() << " clusters from " 
			 << output->size()     << " modules";
  event.put(output);
}
