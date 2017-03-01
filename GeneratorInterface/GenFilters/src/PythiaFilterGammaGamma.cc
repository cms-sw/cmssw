#include "GeneratorInterface/GenFilters/interface/PythiaFilterGammaGamma.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "Math/GenVector/VectorUtil.h"
//#include "CLHEP/HepMC/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "TFile.h"
#include "TLorentzVector.h"

#include <iostream>

using namespace edm;
using namespace std;
using namespace HepMC;


PythiaFilterGammaGamma::PythiaFilterGammaGamma(const edm::ParameterSet& iConfig) :
  token_(consumes<edm::HepMCProduct>(edm::InputTag(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")),"unsmeared"))),
  maxEvents(iConfig.getUntrackedParameter<int>("maxEvents", 100000)),
  ptSeedThr(iConfig.getUntrackedParameter<double>("PtSeedThr")),
  etaSeedThr(iConfig.getUntrackedParameter<double>("EtaSeedThr")),
  ptGammaThr(iConfig.getUntrackedParameter<double>("PtGammaThr")),
  etaGammaThr(iConfig.getUntrackedParameter<double>("EtaGammaThr")),
  ptTkThr(iConfig.getUntrackedParameter<double>("PtTkThr")),
  etaTkThr(iConfig.getUntrackedParameter<double>("EtaTkThr")),
  ptElThr(iConfig.getUntrackedParameter<double>("PtElThr")),
  etaElThr(iConfig.getUntrackedParameter<double>("EtaElThr")),
  dRTkMax(iConfig.getUntrackedParameter<double>("dRTkMax")),
  dRSeedMax(iConfig.getUntrackedParameter<double>("dRSeedMax")),
  dPhiSeedMax(iConfig.getUntrackedParameter<double>("dPhiSeedMax")),
  dEtaSeedMax(iConfig.getUntrackedParameter<double>("dEtaSeedMax")),
  dRNarrowCone(iConfig.getUntrackedParameter<double>("dRNarrowCone")),
  pTMinCandidate1(iConfig.getUntrackedParameter<double>("PtMinCandidate1")),
  pTMinCandidate2(iConfig.getUntrackedParameter<double>("PtMinCandidate2")),
  etaMaxCandidate(iConfig.getUntrackedParameter<double>("EtaMaxCandidate")),
  invMassMin(iConfig.getUntrackedParameter<double>("InvMassMin")),
  invMassMax(iConfig.getUntrackedParameter<double>("InvMassMax")),
  energyCut(iConfig.getUntrackedParameter<double>("EnergyCut")),
  nTkConeMax(iConfig.getUntrackedParameter<int>("NTkConeMax")),
  nTkConeSum(iConfig.getUntrackedParameter<int>("NTkConeSum")),
  acceptPrompts(iConfig.getUntrackedParameter<bool>("AcceptPrompts")), 
  promptPtThreshold(iConfig.getUntrackedParameter<double>("PromptPtThreshold")) {
  
  nSelectedEvents = 0;
  nGeneratedEvents = 0;
  counterPrompt = 0;
  
}

PythiaFilterGammaGamma::~PythiaFilterGammaGamma() 
{  
  cout << "Number of Selected Events: " << nSelectedEvents << endl;
  cout << "Number of Generated Events: " << nGeneratedEvents << endl;
  cout << "Number of Prompt photons: " << counterPrompt << endl;
}

bool PythiaFilterGammaGamma::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
   

  if(nSelectedEvents >= maxEvents) {
    throw cms::Exception("endJob")<<"We have reached the maximum number of events...";
  }

  nGeneratedEvents++;

  bool accepted = false;

  Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);
  myGenEvent = evt->GetEvent();

  std::vector<const GenParticle*> seeds, egamma, stable; 
  std::vector<const GenParticle*>::const_iterator itPart, itStable, itEn;

 // Loop on egamma
  for(HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p) {

    if (
	((*p)->status()==1&&(*p)->pdg_id() == 22) ||                   // gamma
        ((*p)->status()==1&&abs((*p)->pdg_id()) == 11))                // electron

      {       
	// check for eta and pT threshold for seed in gamma, el
	if ((*p)->momentum().perp() > ptSeedThr &&
	    fabs((*p)->momentum().eta()) < etaSeedThr) {
       
	  seeds.push_back(*p);
	}
	}   


    if ((*p)->status() == 1) {

      // save charged stable tracks
      if (abs((*p)->pdg_id()) == 211 ||
          abs((*p)->pdg_id()) == 321 ||
          abs((*p)->pdg_id()) == 11 ||
          abs((*p)->pdg_id()) == 13 ||
          abs((*p)->pdg_id()) == 15) {
        // check if it passes the cut
        if ((*p)->momentum().perp() > ptTkThr &&
            fabs((*p)->momentum().eta()) < etaTkThr) {
          stable.push_back(*p);
        }
      }

      // save egamma for candidate calculation
      if ((*p)->pdg_id() == 22 &&
	  (*p)->momentum().perp() > ptGammaThr &&
	  fabs((*p)->momentum().eta()) < etaGammaThr) {
	egamma.push_back(*p);
      }
      if (abs((*p)->pdg_id()) == 11 &&
	  (*p)->momentum().perp() > ptElThr &&
	  fabs((*p)->momentum().eta()) < etaElThr) {
	egamma.push_back(*p); 
      }
    }
  }

  if (seeds.size() < 2) return accepted;

  std::vector<int> nTracks;
  std::vector<TLorentzVector> candidate, candidateNarrow, candidateSeed;
  std::vector<const GenParticle*>::iterator itSeed;

  const GenParticle* mom;
  int this_id;
  int first_different_id;

  for(itSeed = seeds.begin(); itSeed != seeds.end(); ++itSeed) {

    TLorentzVector energy, narrowCone, temp1, temp2, tempseed;

    tempseed.SetXYZM((*itSeed)->momentum().px(), (*itSeed)->momentum().py(), (*itSeed)->momentum().pz(), 0);   
    for(itEn = egamma.begin(); itEn != egamma.end(); ++itEn) {
      temp1.SetXYZM((*itEn)->momentum().px(), (*itEn)->momentum().py(), (*itEn)->momentum().pz(), 0);        
	
      double DR = temp1.DeltaR(tempseed);
      double DPhi = temp1.DeltaPhi(tempseed);
      double DEta = (*itEn)->momentum().eta()-(*itSeed)->momentum().eta();
      if(DPhi<0) DPhi=-DPhi;
      if(DEta<0) DEta=-DEta;
	
      if (DR < dRSeedMax || (DPhi<dPhiSeedMax&&DEta<dEtaSeedMax)) {
	energy += temp1;
      }
      if (DR < dRNarrowCone) {
	narrowCone += temp1;
      }
    }

    int counter = 0;

    if ( energy.Et() != 0. ) {
      if (fabs(energy.Eta()) < etaMaxCandidate) {

	temp2.SetXYZM(energy.Px(), energy.Py(), energy.Pz(), 0);        
	
	for(itStable = stable.begin(); itStable != stable.end(); ++itStable) {  
	  temp1.SetXYZM((*itStable)->momentum().px(), (*itStable)->momentum().py(), (*itStable)->momentum().pz(), 0);        
	  double DR = temp1.DeltaR(temp2);
	  if (DR < dRTkMax) counter++;        
	}

	if(acceptPrompts) {
	 
	    if ((*itSeed)->momentum().perp()>promptPtThreshold)
	    {
	      bool isPrompt=true;
	      this_id = (*itSeed)->pdg_id();
	      mom = (*itSeed);
	      while (mom->pdg_id() == this_id) {
	   
		const GenParticle* mother = mom->production_vertex() ?       
		  *(mom->production_vertex()->particles_in_const_begin()) : 0;

		mom = mother;
		if (mom == 0) {
		  break;
		}	  
	      }

	      first_different_id = mom->pdg_id();
	  
	      if (mom->status() == 2 && std::abs(first_different_id)>100) isPrompt=false;
	

	      if(isPrompt) counter=0;
	      if(isPrompt) counterPrompt++;
	    }
	}
      }
    }

    candidate.push_back(energy);
    candidateSeed.push_back(tempseed);
    candidateNarrow.push_back(narrowCone);
    nTracks.push_back(counter);
  } 

  if (candidate.size() <2) return accepted;

  TLorentzVector minvMin, minvMax;

  int i1, i2;
  for(unsigned int i=0; i<candidate.size()-1; ++i) {
    
    if (candidate[i].Energy() < energyCut) continue;
    if(nTracks[i]>nTkConeMax) continue;
    if (fabs(candidate[i].Eta()) > etaMaxCandidate) continue;
    
    for(unsigned int j=i+1; j<candidate.size(); ++j) { 
      if (candidate[j].Energy() < energyCut) continue;
      if(nTracks[j]>nTkConeMax) continue;
      if (fabs(candidate[j].Eta()) > etaMaxCandidate) continue;
	  
      if (nTracks[i] + nTracks[j] > nTkConeSum) continue;

      if (candidate[i].Pt() > candidate[j].Pt()) {
	i1 = i;
	i2 = j;
      } 
      else {
	i1 = j;
	i2 = i;
      }

      if (candidate[i1].Pt() < pTMinCandidate1 || candidate[i2].Pt() < pTMinCandidate2) continue;

      minvMin = candidate[i] + candidate[j];
      if (minvMin.M() < invMassMin) continue;
        
      minvMax = candidate[i] + candidate[j];
      if (minvMax.M() > invMassMax) continue;

      accepted = true;

    }
  }
  
  if (accepted) nSelectedEvents++;
  return accepted;
}

