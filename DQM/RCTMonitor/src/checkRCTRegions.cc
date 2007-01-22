#include "DQM/RCTMonitor/src/checkRCTRegions.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
using namespace reco;

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <vector>

#include "TROOT.h"
#include "TNtuple.h"
#include "TFile.h"

// bin 0 is dummy.  absIeta runs 1-32
const float etaLUT[] = {0.0000, 0.0435, 0.1305, 0.2175, 0.3045, 0.3915, 0.4785, 0.5655, 0.6525, 0.7395, 
			0.8265, 0.9135, 1.0005, 1.0875, 1.1745, 1.2615, 1.3485, 1.4355, 1.5225, 1.6095, 
			1.6965, 1.7850, 1.8800, 1.9865, 2.1075, 2.2470, 2.4110, 2.5750, 2.8250, 3.3250, 
			3.8250, 4.3250, 4.825};

const float rctEtaLUT[] = {0.174, 0.522, 0.870, 1.218, 1.566, 1.930, 2.500, 3.325, 3.825, 4.325, 4.825};

checkRCTRegions::checkRCTRegions(const edm::ParameterSet& iConfig)
{
  file = new TFile("checkRCTRegions.root", "RECREATE", "TPG Information");
  file->cd();
  nTuple = new TNtuple("RCTInfo", "RCT Regions nTuple", 
			     "nEcalDigi:ecalSum:ecalMax:nHcalDigi:hbSum:hbMax:heSum:heMax:hfSum:hfMax:etSum:id:st:pt:eta:phi:towerEtSum:hcalTowerEtSum:regionEtSum:regionEta:regionPhi:regionTauVeto:regionMIPBit:regionQuietBit");
}

checkRCTRegions::~checkRCTRegions()
{
  file->cd();
  nTuple->Write();
  file->Close();
}


void checkRCTRegions::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<CandidateCollection> genParticlesHandle;
  iEvent.getByLabel( "genParticleCandidates", genParticlesHandle);
  CandidateCollection genParticles = *genParticlesHandle;
  edm::ESHandle<CaloTPGTranscoder> transcoder;
  iSetup.get<CaloTPGRecord>().get(transcoder);
  edm::Handle<EcalTrigPrimDigiCollection> ecal;
  edm::Handle<HcalTrigPrimDigiCollection> hcal;
  iEvent.getByType(ecal);
  iEvent.getByLabel("hcalTriggerPrimitiveDigis",hcal);
  EcalTrigPrimDigiCollection ecalCollection = *ecal;
  HcalTrigPrimDigiCollection hcalCollection = *hcal;
  edm::Handle<L1CaloRegionCollection> rctRegions;
  iEvent.getByType(rctRegions);
  int nEcalDigi = ecalCollection.size();
  if (nEcalDigi>4032) {nEcalDigi=4032;}
  float ecalSum = 0;
  float ecalMax = 0;
  for (int i = 0; i < nEcalDigi; i++){
    unsigned short energy = ecalCollection[i].compressedEt();
    float et = float(energy)/2.;  // Temporarily ET is hardcoded to be in 0.5 GeV steps in linear scale
    if(energy > 3) ecalSum += et;    // Make sure that there is a 3*LSB cut
    if(et > ecalMax) ecalMax = et;
  }
  float hbSum = 0;
  float heSum = 0;
  float hfSum = 0;
  float hbMax = 0;
  float heMax = 0;
  float hfMax = 0;
  int nHcalDigi = hcalCollection.size();
  if (nHcalDigi != 4176){ cerr << "There are " << nHcalDigi << " instead of 4176!" << endl;}
  // incl HF 4032 + 144 = 4176
  for (int i = 0; i < nHcalDigi; i++){
    short ieta = (short) hcalCollection[i].id().ieta(); 
    unsigned short absIeta = (unsigned short) abs(ieta);
    unsigned short energy = hcalCollection[i].SOI_compressedEt();     // access only sample of interest
    float et = transcoder->hcaletValue(absIeta, energy);
    if (absIeta <= 28){
      if(absIeta <= 21) {
	if(energy > 3) hbSum += et;   // Make sure that there is a 3*LSB cut
	if(et > hbMax) hbMax = et;
      }
      if(absIeta > 21) {
	if(energy > 3) heSum += et;   // Make sure that there is a 3*LSB cut
	if(et > heMax) heMax = et;
      }
    }
    else if ((absIeta >= 29) && (absIeta <= 32)){
      if(energy > 3) hfSum += et;   // Make sure that there is a 3*LSB cut
      if(et > hfMax) hfMax = et;
    }
  }
  int id = 0;
  int st = 0;
  float pt = 0;
  float phi = 0;
  float eta = 0;
  float towerEtSum = 0;
  float hcalTowerEtSum = 0.;
  float regionEtSum = 0;
  float regionEtaEtSum = 0;
  float regionPhiEtSum = 0;
  bool regionTauVeto = false;
  bool regionMIPBit = true;
  bool regionQuietBit = true;
  for(size_t i = 0; i < genParticles.size(); ++ i ) {
    const Candidate & p = genParticles[ i ];
    // For maximum PT stable particle in the detector
    if((status(p) == 1) && (pt < p.pt()) && (p.eta() > -5) && (p.eta() < 5))
      {
	id = pdgId( p );
	st = status( p );
	pt = p.pt();
	eta = p.eta();
	phi = p.phi();
	towerEtSum = 0;
	for (int i = 0; i < nEcalDigi; i++){
	  unsigned short energy = ecalCollection[i].compressedEt();
	  float towerEt = float(energy)/2.;  // Temporarily ET is hardcoded to be in 0.5 GeV steps in linear scale
	  short ieta = (short) ecalCollection[i].id().ieta(); 
	  unsigned short absIeta = (unsigned short) abs(ieta);    // absIeta runs 1-32
	  float towerEta = (ieta / absIeta) * etaLUT[absIeta];    // Lookup bin centers
	  short cal_iphi = ecalCollection[i].id().iphi();
	  float towerPhi = float(cal_iphi) * 3.1415927 / 36.;
	  if(towerPhi > 3.1415927) towerPhi -= (2 * 3.1415927);
	  float deltaPhi = phi - towerPhi;
	  if(deltaPhi > (2 * 3.1415827)) deltaPhi -= (2 * 3.1415927);
	  float deltaRSq = (eta - towerEta) * (eta - towerEta) + deltaPhi * deltaPhi;
	  if(deltaRSq < 0.5 * 0.5)
	    {
	      towerEtSum += towerEt;
	    }
	  if(energy > 5)
	    {
	      cout << "ECAL TPG: (Et,iEta,iPhi)=(" << float(energy)/2. << "," 
		   << ieta << "," << cal_iphi << ")" << endl;
	    }
	}
	hcalTowerEtSum = 0.;
	for (int i = 0; i < nHcalDigi; i++){
	  unsigned short energy = hcalCollection[i].SOI_compressedEt();     // access only sample of interest
	  short ieta = (short) hcalCollection[i].id().ieta(); 
	  unsigned short absIeta = (unsigned short) abs(ieta);
	  float towerEt = transcoder->hcaletValue(absIeta, energy);
	  float towerEta = (ieta / absIeta) * etaLUT[absIeta];    // Lookup bin centers
	  short cal_iphi = hcalCollection[i].id().iphi();
	  float towerPhi = float(cal_iphi) * 3.1415927 / 36.;
	  if(towerPhi > 3.1415927) towerPhi -= (2 * 3.1415927);
	  float deltaPhi = phi - towerPhi;
	  if(deltaPhi > (2 * 3.1415827)) deltaPhi -= (2 * 3.1415927);
	  float deltaRSq = (eta - towerEta) * (eta - towerEta) + deltaPhi * deltaPhi;
	  if(deltaRSq < 0.5 * 0.5)
	    {
	      towerEtSum += towerEt;
	      hcalTowerEtSum += towerEt;
	    }
	  if(towerEt > 5)
	    {
	      cout << "HCAL TPG: (Et,iEta,iPhi)=(" << towerEt << "," 
		   << ieta << "," << cal_iphi << ")" << endl;
	    }
	}
	int nRCTRegions = (*rctRegions).size();
	L1CaloRegionCollection::const_iterator region;
	for (region=rctRegions->begin(); region!=rctRegions->end(); region++){
	  unsigned regionEt = region->et();
	  unsigned rctCrate = region->rctCrate();
	  float regionEta = 999.;
	  if(rctCrate < 9) regionEta = -rctEtaLUT[region->rctEta()];
	  else regionEta = rctEtaLUT[region->rctEta()];
	  float deltaEta = eta - regionEta;
	  unsigned rctIPhi;
	  if(rctCrate < 9) rctIPhi = rctCrate * 2 + region->rctPhi();
	  else rctIPhi = (rctCrate - 9) * 2 + region->rctPhi();
	  float regionPhi = (3.1415927 / 2.) + 2 * 0.087 - float(rctIPhi) * 2. * 3.1415927 / 18.;
	  if(regionPhi < -3.1415927) regionPhi += (2 * 3.1415927);
	  float deltaPhi = phi - regionPhi;
	  if(deltaPhi > (2 * 3.1415827)) deltaPhi -= (2 * 3.1415927);
	  float deltaRSq = deltaEta * deltaEta + deltaPhi * deltaPhi;
	  if(deltaRSq < 0.5 * 0.5) 
	    {
	      regionEtSum += regionEt;
	      if(region->tauVeto()) regionTauVeto = true;
	      if(!region->mip()) regionMIPBit = false;
	      if(!region->quiet()) regionQuietBit = false;
	      regionEtaEtSum += regionEta * regionEt;
	      regionPhiEtSum += regionPhi * regionEt;
	    }
	  if(regionEt > 20)   // More than 10 GeV at the 0.5 GeV LSB default
	    {
	      cout << "(RCT Region Dump)\n" << (*region) << endl;
	      cout << "\nregion(Et,Eta,Phi)=(" 
		   << regionEt << ","
		   << regionEta << ","
		   << regionPhi << ")" 
		   << "\tDelta(Et,Eta,Phi)=(" 
		   << (pt - regionEt) << ","
		   << (eta - regionEta) << ","
		   << deltaPhi << ")" 
		   << endl;
	    }
	}
      }
  }
  std::vector<float> result;
  result.push_back(nEcalDigi);
  result.push_back(ecalSum);
  result.push_back(ecalMax);
  result.push_back(nHcalDigi);
  result.push_back(hbSum);
  result.push_back(hbMax);
  result.push_back(heSum);
  result.push_back(heMax);
  result.push_back(hfSum);
  result.push_back(hfMax);
  result.push_back(ecalSum+hbSum+heSum+hfSum);
  result.push_back(id);
  result.push_back(st);
  result.push_back(pt);
  result.push_back(eta);
  result.push_back(phi);
  result.push_back(towerEtSum);
  result.push_back(hcalTowerEtSum);
  result.push_back(regionEtSum);
  result.push_back(regionEtaEtSum/regionEtSum);
  result.push_back(regionPhiEtSum/regionEtSum);
  result.push_back(regionTauVeto);
  result.push_back(regionMIPBit);
  result.push_back(regionQuietBit);
  nTuple->Fill(&result[0]);  // Assumes vector is implemented internally as an array
}
