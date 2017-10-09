#ifndef EventFilter_EcalRecHitsMerger_H
#define EventFilter_EcalRecHitsMerger_H

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

class EcalRecHitsMerger : public edm::EDProducer {

public:
	EcalRecHitsMerger(const edm::ParameterSet& pset);
	virtual ~EcalRecHitsMerger();
	void produce(edm::Event & e, const edm::EventSetup& c) override;
	void beginJob(void) override;
	void endJob(void) override;
	static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
	edm::InputTag EgammaSourceEB_;
   	edm::InputTag MuonsSourceEB_ ;
	edm::InputTag TausSourceEB_ ;
	edm::InputTag JetsSourceEB_ ;
	edm::InputTag RestSourceEB_ ;
	edm::InputTag Pi0SourceEB_ ;
	std::string OutputLabelEB_;

        edm::InputTag EgammaSourceEE_;
        edm::InputTag MuonsSourceEE_ ;
	edm::InputTag TausSourceEE_ ;
        edm::InputTag JetsSourceEE_ ;
	edm::InputTag RestSourceEE_ ;
	edm::InputTag Pi0SourceEE_ ;
        std::string OutputLabelEE_;
	
	std::string InputRecHitEB_;
	std::string InputRecHitEE_;
	

	bool debug_ ;

};

#endif


