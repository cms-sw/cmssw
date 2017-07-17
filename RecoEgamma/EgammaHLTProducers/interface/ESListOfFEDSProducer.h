#ifndef EventFilter_ESListOfFEDSProducer_H
#define EventFilter_ESListOfFEDSProducer_H

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>
                                                                                             
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/ESElectronicsMapper.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

#include <iostream>
#include <string>
#include <vector>

namespace edm {
  class ConfigurationDescriptions;
}

class ESListOfFEDSProducer : public edm::EDProducer {

public:
	ESListOfFEDSProducer(const edm::ParameterSet& pset);
	virtual ~ESListOfFEDSProducer();
	void produce(edm::Event & e, const edm::EventSetup& c);
	void beginJob(void);
	void endJob(void);
	void Egamma(edm::Event& e, const edm::EventSetup& es, std::vector<int>& done, std::vector<int>& FEDs);
	void Muon(edm::Event& e, const edm::EventSetup& es, std::vector<int>& done, std::vector<int>& FEDs);
	void Jets(edm::Event& e, const edm::EventSetup& es, std::vector<int>& done, std::vector<int>& FEDs);
	static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
	//edm::InputTag Pi0ListToIgnore_; 
	bool EGamma_;
	edm::EDGetTokenT<l1extra::L1EmParticleCollection> EMl1TagIsolated_;
	edm::EDGetTokenT<l1extra::L1EmParticleCollection> EMl1TagNonIsolated_;
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
	edm::EDGetTokenT<l1extra::L1MuonParticleCollection> MuonSource_ ;

	bool Jets_ ;
	bool JETSdoCentral_ ;
	bool JETSdoForward_ ;
	bool JETSdoTau_ ;
	double JETSregionEtaMargin_;
	double JETSregionPhiMargin_;
	double Ptmin_jets_ ;
	edm::EDGetTokenT<l1extra::L1JetParticleCollection> CentralSource_;
        edm::EDGetTokenT<l1extra::L1JetParticleCollection> ForwardSource_;
	edm::EDGetTokenT<l1extra::L1JetParticleCollection> TauSource_;

	std::string OutputLabel_;
	EcalElectronicsMapping* TheMapping;

	///	ESElectronicsMapping* TheESMapping;
	ESElectronicsMapper *TheESMapping;
	
	bool first_ ;
	bool debug_ ;

	std::vector<int> ListOfFEDS(double etaLow, double etaHigh, double phiLow,
                                    double phiHigh, double etamargin, double phimargin);



};

#endif


