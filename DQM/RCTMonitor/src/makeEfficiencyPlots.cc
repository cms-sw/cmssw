#include "DQM/RCTMonitor/src/makeEfficiencyPlots.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
using namespace reco;

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

makeEfficiencyPlots::makeEfficiencyPlots(const edm::ParameterSet& iConfig)
{
}

makeEfficiencyPlots::~makeEfficiencyPlots()
{
}


void makeEfficiencyPlots::analyze(const edm::Event& event, const edm::EventSetup& iSetup)
{
  edm::Handle<CandidateCollection> genParticlesHandle;
  event.getByLabel( "genParticleCandidates", genParticlesHandle);
  CandidateCollection genParticles = *genParticlesHandle;
  for(size_t i = 0; i < genParticles.size(); ++ i ) {
    const Candidate & p = genParticles[ i ];
    int id = pdgId( p );
    int st = status( p );
    double pt = p.pt(), eta = p.eta(), phi = p.phi(), mass = p.mass();
    double vx = p.vx(), vy = p.vy(), vz = p.vz();
    int charge = p.charge();
    int n = p.numberOfDaughters();
    cout << "Found particle: " << id << " with status " << st << " pt=" << pt << " eta=" << eta << " phi=" << phi
	 << " mass=" << mass << " charge=" << charge << endl;
 }
}
