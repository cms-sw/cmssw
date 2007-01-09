#include "DQM/RCTMonitor/src/checkTPGs.h"

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

// bin 0 is dummy.  absIeta runs 1-32
const float etaLUT[] = {0.0000, 0.0435, 0.1305, 0.2175, 0.3045, 0.3915, 0.4785, 0.5655, 0.6525, 0.7395, 
			0.8265, 0.9135, 1.0005, 1.0875, 1.1745, 1.2615, 1.3485, 1.4355, 1.5225, 1.6095, 
			1.6965, 1.7850, 1.8800, 1.9865, 2.1075, 2.2470, 2.4110, 2.5750, 2.8250, 3.3250, 
			3.8250, 4.3250, 4.825};

checkTPGs::checkTPGs(const edm::ParameterSet& iConfig)
{
}

checkTPGs::~checkTPGs()
{
}


void checkTPGs::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::ESHandle<CaloTPGTranscoder> transcoder;
  iSetup.get<CaloTPGRecord>().get(transcoder);
  edm::Handle<EcalTrigPrimDigiCollection> ecal;
  edm::Handle<HcalTrigPrimDigiCollection> hcal;
  iEvent.getByType(ecal);
  iEvent.getByLabel("hcalTriggerPrimitiveDigis",hcal);
  EcalTrigPrimDigiCollection ecalCollection = *ecal;
  HcalTrigPrimDigiCollection hcalCollection = *hcal;
  int nEcalDigi = ecalCollection.size();
  if (nEcalDigi>4032) {nEcalDigi=4032;}
  float ecalSum = 0;
  float ecalMax = 0;
  for (int i = 0; i < nEcalDigi; i++){
    unsigned short energy = ecalCollection[i].compressedEt();
    float et = float(energy)/2.;  // Temporarily ET is hardcoded to be in 0.5 GeV steps in linear scale
    if(et > 1) ecalSum += et;
    if(et > ecalMax) ecalMax = et;
  }
  float hbSum = 0;
  float heSum = 0;
  float hfSum = 0;
  float hbMax = 0;
  float heMax = 0;
  float hfMax = 0;
  int nHcalDigi = hcalCollection.size();
  if (nHcalDigi != 4176){ cout << "There are " << nHcalDigi << " instead of 4176!" << endl;}
  // incl HF 4032 + 144 = 4176
  for (int i = 0; i < nHcalDigi; i++){
    short ieta = (short) hcalCollection[i].id().ieta(); 
    unsigned short absIeta = (unsigned short) abs(ieta);
    unsigned short energy = hcalCollection[i].SOI_compressedEt();     // access only sample of interest
    float et = transcoder->hcaletValue(absIeta, energy);
    if (absIeta <= 28){
      if(absIeta <= 24) {
	hbSum += et;
	if(et > hbMax) hbMax = et;
      }
      if(absIeta > 24) {
	heSum += et;
	if(et > heMax) heMax = et;
      }
    }
    else if ((absIeta >= 29) && (absIeta <= 32)){
      hfSum += et;
      if(et > hfMax) hfMax = et;
    }
  }
  cout << endl;
  cout << "ecalSum = " << ecalSum << "\thbSum = " << hbSum << "\theSum = " << heSum << "\thfSum = " << hfSum << endl;
  cout << "ecalMax = " << ecalMax << "\thbMax = " << hbMax << "\theMax = " << heMax << "\thfMax = " << hfMax << endl;
  cout << endl;
  edm::Handle<CandidateCollection> genParticlesHandle;
  iEvent.getByLabel( "genParticleCandidates", genParticlesHandle);
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
    float towerEtSum = 0;
    for (int i = 0; i < nEcalDigi; i++){
      unsigned short energy = ecalCollection[i].compressedEt();
      float towerEt = float(energy)/2.;  // Temporarily ET is hardcoded to be in 0.5 GeV steps in linear scale
      short ieta = (short) ecalCollection[i].id().ieta(); 
      unsigned short absIeta = (unsigned short) abs(ieta);    // absIeta runs 1-32
      float towerEta = (ieta / absIeta) * etaLUT[absIeta];    // Lookup bin centers
      unsigned short cal_iphi = (unsigned short) ecalCollection[i].id().iphi();
      float towerPhi = float(cal_iphi) * 3.1415927 / 36.;
      if(towerPhi > 3.1415927) towerPhi -= 3.1415927;
      if(towerEt > 2.)
	{
	  cout << "ECAL towerEt=" << towerEt << " towerEta=" << towerEta << " towerPhi=" << towerPhi << endl;
	}
      float deltaRSq = (eta - towerEta) * (eta - towerEta) + (phi - towerPhi) * (phi - towerPhi);
      if(deltaRSq < 0.1 * 0.1)
	{
	  towerEtSum += towerEt;
	}
    }
    float hcalTowerEtSum = 0.;
    for (int i = 0; i < nHcalDigi; i++){
      unsigned short energy = hcalCollection[i].SOI_compressedEt();     // access only sample of interest
      short ieta = (short) hcalCollection[i].id().ieta(); 
      unsigned short absIeta = (unsigned short) abs(ieta);
      float towerEt = transcoder->hcaletValue(absIeta, energy);
      float towerEta = (ieta / absIeta) * etaLUT[absIeta];    // Lookup bin centers
      unsigned short cal_iphi = (unsigned short) hcalCollection[i].id().iphi();
      float towerPhi = float(cal_iphi) * 3.1415927 / 36.;
      if(towerPhi > 3.1415927) towerPhi -= 3.1415927;
      if(towerEt > 2.)
	{
	  cout << "HCAL towerEt=" << towerEt << " towerEta=" << towerEta << " towerPhi=" << towerPhi << endl;
	}
      float deltaRSq = (eta - towerEta) * (eta - towerEta) + (phi - towerPhi) * (phi - towerPhi);
      if(deltaRSq < 0.1 * 0.1)
	{
	  towerEtSum += towerEt;
	  hcalTowerEtSum += towerEt;
	}
    }
    cout << "Trigger tower primitives neighborhood sum ET =" << towerEtSum;
    if(towerEtSum > 0)
      cout << " HCAL fraction =" << (100. * hcalTowerEtSum / towerEtSum) << "%" << endl;
    else
      cout << endl;
  }
}
