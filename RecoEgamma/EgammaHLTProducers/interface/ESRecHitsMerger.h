#ifndef EventFilter_ESRecHitsMerger_H
#define EventFilter_ESRecHitsMerger_H

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>
                                                                                             
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <iostream>
#include <string>
#include <vector>

namespace edm {
  class ConfigurationDescriptions;
}

class ESRecHitsMerger : public edm::EDProducer {

public:
	ESRecHitsMerger(const edm::ParameterSet& pset);
	virtual ~ESRecHitsMerger();
	void produce(edm::Event & e, const edm::EventSetup& c);
	void beginJob(void);
	void endJob(void);
	static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
	edm::InputTag EgammaSourceES_;
   	edm::InputTag MuonsSourceES_ ;
	edm::InputTag TausSourceES_ ;
	edm::InputTag JetsSourceES_ ;
	edm::InputTag RestSourceES_ ;
	edm::InputTag Pi0SourceES_ ;
	edm::InputTag EtaSourceES_ ;
	std::string OutputLabelES_;

	std::string InputRecHitES_;
	

	bool debug_ ;

};

#endif


