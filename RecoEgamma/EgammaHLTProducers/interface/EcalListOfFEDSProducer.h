#ifndef EventFilter_EcalListOfFEDSProducer_H
#define EventFilter_EcalListOfFEDSProducer_H

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>
                                                                                             
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
                                                                                             
#include <iostream>
#include <string>
#include <vector>

class EcalListOfFEDSProducer : public edm::EDProducer {

public:
	EcalListOfFEDSProducer(const edm::ParameterSet& pset);
	virtual ~EcalListOfFEDSProducer();
	void produce(edm::Event & e, const edm::EventSetup& c);
	void beginJob(const edm::EventSetup& c);
	void endJob(void);
	std::vector<int> Egamma(edm::Event& e, const edm::EventSetup& es);
	std::vector<int> Muon(edm::Event& e, const edm::EventSetup& es);

private:
	bool EGamma_;
	edm::InputTag EMl1TagIsolated_;
	edm::InputTag EMl1TagNonIsolated_;
	bool EMdoIsolated_;
	bool EMdoNonIsolated_;
	double EMregionEtaMargin_;
	double EMregionPhiMargin_;
	double Ptmin_iso_ ;
	double Ptmin_noniso_;

	bool Muon_ ;
	double MUregionEtaMargin_;
	double MUregionPhiMargin_;
	double Ptmin_muon_ ;
	edm::InputTag MuonSource_ ;

	std::string OutputLabel_;
	EcalElectronicsMapping* TheMapping;
	bool first_ ;
	bool debug_ ;

	std::vector<int> ListOfFEDS(double etaLow, double etaHigh, double phiLow,
                                    double phiHigh, double etamargin, double phimargin);



};

#endif


