#include "GeneratorInterface/Core/interface/PythiaHepMCFilterGammaGamma.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "Math/GenVector/VectorUtil.h"
//#include "CLHEP/HepMC/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "TLorentzVector.h"

#include <iostream>

using namespace edm;
using namespace std;
using namespace HepMC;


PythiaHepMCFilterGammaGamma::PythiaHepMCFilterGammaGamma(const edm::ParameterSet& iConfig) :
  ptSeedThr(iConfig.getParameter<double>("PtSeedThr")),
  etaSeedThr(iConfig.getParameter<double>("EtaSeedThr")),
  ptGammaThr(iConfig.getParameter<double>("PtGammaThr")),
  etaGammaThr(iConfig.getParameter<double>("EtaGammaThr")),
  ptTkThr(iConfig.getParameter<double>("PtTkThr")),
  etaTkThr(iConfig.getParameter<double>("EtaTkThr")),
  ptElThr(iConfig.getParameter<double>("PtElThr")),
  etaElThr(iConfig.getParameter<double>("EtaElThr")),
  dRTkMax(iConfig.getParameter<double>("dRTkMax")),
  dRSeedMax(iConfig.getParameter<double>("dRSeedMax")),
  dPhiSeedMax(iConfig.getParameter<double>("dPhiSeedMax")),
  dEtaSeedMax(iConfig.getParameter<double>("dEtaSeedMax")),
  dRNarrowCone(iConfig.getParameter<double>("dRNarrowCone")),
  pTMinCandidate1(iConfig.getParameter<double>("PtMinCandidate1")),
  pTMinCandidate2(iConfig.getParameter<double>("PtMinCandidate2")),
  etaMaxCandidate(iConfig.getParameter<double>("EtaMaxCandidate")),
  invMassMin(iConfig.getParameter<double>("InvMassMin")),
  invMassMax(iConfig.getParameter<double>("InvMassMax")),
  energyCut(iConfig.getParameter<double>("EnergyCut")),
  nTkConeMax(iConfig.getParameter<int>("NTkConeMax")),
  nTkConeSum(iConfig.getParameter<int>("NTkConeSum")),
  acceptPrompts(iConfig.getParameter<bool>("AcceptPrompts")),
  promptPtThreshold(iConfig.getParameter<double>("PromptPtThreshold")) {

}

PythiaHepMCFilterGammaGamma::~PythiaHepMCFilterGammaGamma() 
{  
}

bool PythiaHepMCFilterGammaGamma::filter(const HepMC::GenEvent* myGenEvent) {
   
  bool accepted = false;

  // electron/photon seeds
  std::vector<const GenParticle*> seeds;

  // other electrons/photons to be added to seeds
  // to form candidates
  std::vector<const GenParticle*> egamma;

  // charged tracks to be taken into account in the isolation cones
  // around candidates
  std::vector<const GenParticle*> stable;

  std::vector<const GenParticle*>::const_iterator itPart, itStable, itEn;

  //----------
  // 1. find electron/photon seeds
  //----------
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
      if (abs((*p)->pdg_id()) == 211 || // charged pion
          abs((*p)->pdg_id()) == 321 || // charged kaon
          abs((*p)->pdg_id()) == 11 ||  // electron
          abs((*p)->pdg_id()) == 13 ||  // muon
          abs((*p)->pdg_id()) == 15) {  // tau
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

  //----------
  // 2. loop over seeds to build candidates
  //
  //    (adding nearby electrons/photons
  //     to the seed electrons/photons to obtain the total
  //     electromagnetic energy)
  //----------

  // number of tracks around each of the candidates
  std::vector<int> nTracks;

  // the candidates (four momenta) formed from the
  // seed electrons/photons and nearby electrons/photons
  std::vector<TLorentzVector> candidate;

  // these are filled but then not used afterwards (could be removed)
  std::vector<TLorentzVector> candidateNarrow, candidateSeed;

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

      // accept if within cone or within rectangular region around seed
      if (DR < dRSeedMax || (DPhi<dPhiSeedMax&&DEta<dEtaSeedMax)) {
	energy += temp1;
      }
      if (DR < dRNarrowCone) {
	narrowCone += temp1;
      }
    }

    // number of stable charged particles found within dRTkMax
    // around candidate
    int counter = 0;

    if ( energy.Et() != 0. ) {
      if (fabs(energy.Eta()) < etaMaxCandidate) {

	temp2.SetXYZM(energy.Px(), energy.Py(), energy.Pz(), 0);        

        // count number of stable particles within cone around candidate
	for(itStable = stable.begin(); itStable != stable.end(); ++itStable) {  
	  temp1.SetXYZM((*itStable)->momentum().px(), (*itStable)->momentum().py(), (*itStable)->momentum().pz(), 0);        
	  double DR = temp1.DeltaR(temp2);
	  if (DR < dRTkMax) counter++;        
	}

	if(acceptPrompts) {
	 
	    if ((*itSeed)->momentum().perp()>promptPtThreshold)
	    {
	      // check if *itSeed is a prompt particle

	      bool isPrompt=true;
	      this_id = (*itSeed)->pdg_id();
	      mom = (*itSeed);
	      while (mom->pdg_id() == this_id) {
	   
		const GenParticle* mother = mom->production_vertex() ?       
		  *(mom->production_vertex()->particles_in_const_begin()) : nullptr;

		mom = mother;
		if (mom == nullptr) {
		  break;
		}	  
	      }

	      first_different_id = mom->pdg_id();
	  
	      if (mom->status() == 2 && std::abs(first_different_id)>100) isPrompt=false;
	
	      // ignore charged particles around prompt particles
	      if(isPrompt) counter=0;
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

  //----------
  // 3. perform further checks on candidates
  //
  //    (energy, charged isolation requirements etc.)
  //----------

  int i1, i2;
  for(unsigned int i=0; i<candidate.size()-1; ++i) {
    
    if (candidate[i].Energy() < energyCut) continue;
    if(nTracks[i]>nTkConeMax) continue;
    if (fabs(candidate[i].Eta()) > etaMaxCandidate) continue;
    
    for(unsigned int j=i+1; j<candidate.size(); ++j) { 

      // check features of second candidate alone
      if (candidate[j].Energy() < energyCut) continue;
      if(nTracks[j]>nTkConeMax) continue;
      if (fabs(candidate[j].Eta()) > etaMaxCandidate) continue;
	  
      // check requirement on sum of tracks in both isolation cones
      if (nTracks[i] + nTracks[j] > nTkConeSum) continue;

      // swap candidates to have pt[i1] >= pt[i2]
      if (candidate[i].Pt() > candidate[j].Pt()) {
	i1 = i;
	i2 = j;
      } 
      else {
	i1 = j;
	i2 = i;
      }

      // require minimum pt on leading and subleading candidate
      if (candidate[i1].Pt() < pTMinCandidate1 || candidate[i2].Pt() < pTMinCandidate2) continue;

      // apply requirements on candidate pair mass
      minvMin = candidate[i] + candidate[j];
      if (minvMin.M() < invMassMin) continue;
        
      minvMax = candidate[i] + candidate[j];
      if (minvMax.M() > invMassMax) continue;

      accepted = true;

    }
  }

  return accepted;
}

