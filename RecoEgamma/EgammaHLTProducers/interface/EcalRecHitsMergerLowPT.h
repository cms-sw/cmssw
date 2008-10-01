#ifndef EventFilter_EcalRecHitsMergerLowPT_H
#define EventFilter_EcalRecHitsMergerLowPT_H

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

class EcalRecHitsMergerLowPT : public edm::EDProducer {

public:
	EcalRecHitsMergerLowPT(const edm::ParameterSet& pset);
	virtual ~EcalRecHitsMergerLowPT();
	void produce(edm::Event & e, const edm::EventSetup& c);
	void beginJob(const edm::EventSetup& c);
	void endJob(void);

private:
	edm::InputTag MergedSourceEB_;

	edm::InputTag LowPTSourceEB_;
	edm::InputTag LowPTJetSourceEB_;
	edm::InputTag JetsSourceEB_;
	edm::InputTag JetsSourceEE_;
	edm::InputTag TausSourceEB_;
	edm::InputTag TausSourceEE_;


	std::string OutputLabelEB_;

        edm::InputTag MergedSourceEE_;
	edm::InputTag LowPTSourceEE_;
	edm::InputTag LowPTJetSourceEE_;

        std::string OutputLabelEE_;
	
	std::string InputRecHitEB_;
	std::string InputRecHitEE_;
	

	edm::InputTag LowPTEgammaSourceEB_;
	edm::InputTag LowPTEgammaSourceEE_;
	
	bool debug_ ;

};

#endif


