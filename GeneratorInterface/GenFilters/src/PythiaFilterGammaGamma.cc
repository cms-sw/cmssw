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
  label(iConfig.getUntrackedParameter<std::string>("moduleLabel",std::string("generator"))),
  //fileName(iConfig.getUntrackedParameter<std::string>("fileName", std::string("plots.root"))),
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
  invMassWide(iConfig.getUntrackedParameter<double>("InvMassWide")),
  invMassNarrow(iConfig.getUntrackedParameter<double>("InvMassNarrow")),
  nTkConeMax(iConfig.getUntrackedParameter<int>("NTkConeMax")),
  nTkConeSum(iConfig.getUntrackedParameter<int>("NTkConeSum")),
  acceptPrompts(iConfig.getUntrackedParameter<bool>("AcceptPrompts")), 
  promptPtThreshold(iConfig.getUntrackedParameter<double>("PromptPtThreshold")) {

  //cout<<"ptSeedThr "<<ptSeedThr<<endl;
  //cout<<"etaSeedThr "<<etaSeedThr<<endl;
  //cout<<"ptGammaThr "<<ptGammaThr<<endl;
  //cout<<"etaGammaThr "<<etaGammaThr<<endl;
  //cout<<"ptTkThr "<<ptTkThr<<endl;
  //cout<<"etaTkThr "<<etaTkThr<<endl;
  //cout<<"ptElThr "<<ptElThr<<endl;
  //cout<<"etaElThr "<<etaElThr<<endl;
  //cout<<"dRTkMax "<<dRTkMax<<endl;
  //cout<<"dRSeedMax "<<dRSeedMax<<endl;
  //cout<<"dPhiSeedMax "<<dPhiSeedMax<<endl;
  //cout<<"dEtaSeedMax "<<dEtaSeedMax<<endl;
  //cout<<"dRNarrowCone "<<dRNarrowCone<<endl;
  //cout<<"pTMinCandidate1 "<<pTMinCandidate1<<endl;
  //cout<<"pTMinCandidate2 "<<pTMinCandidate2<<endl;
  //cout<<"etaMaxCandidate "<<etaMaxCandidate<<endl;
  //cout<<"invMassWide "<<invMassWide<<endl;
  //cout<<"invMassNarrow "<<invMassNarrow<<endl;
  //cout<<"nTkConeMax "<<nTkConeMax<<endl;
  //cout<<"nTkConeSum "<<nTkConeSum<<endl;
  //cout<<"acceptPrompts "<<acceptPrompts<<endl;
  //cout<<"promptPtThreshold "<<promptPtThreshold<<endl;
  
  nSelectedEvents = 0;
  nGeneratedEvents = 0;

  //char a[20];
  //for(int i=0; i<2; i++) {
  //  sprintf(a, "PT Seed %d", i+1);
  //  hPtSeed[i] = new TH1D(a, a, 100, 0, 200);
  //  sprintf(a, "Eta Seed %d", i+1);
  //  hEtaSeed[i]= new TH1D(a, a, 100, -3., 3.);
  //  sprintf(a, "Pid Seed %d", i+1);
  //  hPidSeed[i]= new TH1I(a, a, 50, 0, 500); 
  //  sprintf(a, "PT Candidate %d", i+1);
  //  hPtCandidate[i] = new TH1D(a, a, 100, 0, 200);
  //  sprintf(a, "Eta Candidate %d", i+1);
  //  hEtaCandidate[i]= new TH1D(a, a, 100, -3., 3.);
  //  sprintf(a, "Pid Candidate %d", i+1);
  //  hPidCandidate[i]= new TH1I(a, a, 50, 0, 500);
  //  sprintf(a, "NTk Iso %d", i+1);
  //  hNTk[i]= new TH1I(a, a, 50, 0, 50);
  //}
  //hMassNarrow = new TH1D("Mass narr.", "Mass narr.", 100, 0, 1000);
  //hMassWide= new TH1D("Mass wide", "Mass wide", 100, 0, 1000);
  //hNTkSum= new TH1I("NTk Sum Iso", "NTk Sum Iso", 50, 0, 50);
  
}

PythiaFilterGammaGamma::~PythiaFilterGammaGamma() 
{  
  //writeFile();
  cout << "Number of Selected Events: " << nSelectedEvents << endl;
  cout << "Number of Generated Events: " << nGeneratedEvents << endl;
}

//void PythiaFilterGammaGamma::writeFile() {

  //TFile* file = new TFile (fileName.c_str(), "recreate");
  
  //for(int i=0; i<2; i++) {
  //  hPtSeed[i]->Write();
  //  hEtaSeed[i]->Write();
  //  hPidSeed[i]->Write(); 
  //  hPtCandidate[i]->Write();
  //  hEtaCandidate[i]->Write();
  //  hPidCandidate[i]->Write();
  //  hNTk[i]->Write();
  //}
  //hMassNarrow->Write();
  //hMassWide->Write();
  //hNTkSum->Write();
  
  //file->Close();
  
//}

bool PythiaFilterGammaGamma::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  if(nSelectedEvents >= maxEvents) {
    //writeFile();
    throw cms::Exception("endJob")<<"We have reached the maximum number of events...";
  }

  nGeneratedEvents++;

  bool accepted = false;

  Handle<HepMCProduct> evt;
  iEvent.getByLabel(label, evt);
  myGenEvent = evt->GetEvent();

  std::vector<const GenParticle*> seeds, egamma, stable; 
  std::vector<const GenParticle*>::const_iterator itPart, itStable, itEn;

  for(HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p) {

    if (
	((*p)->status()==1&&(*p)->pdg_id() == 22) ||                   // gamma
        ((*p)->status()==1&&abs((*p)->pdg_id()) == 11) ||              // electron
        (*p)->pdg_id() == 111 ||                  // pi0
        abs((*p)->pdg_id()) == 221 ||             // eta
        abs((*p)->pdg_id()) == 331 ||             // eta prime
        abs((*p)->pdg_id()) == 113 ||             // rho0 
        abs((*p)->pdg_id()) == 223)               // omega
      {       
	// check for eta and pT threshold for seed in gamma, el
	if ((*p)->momentum().perp() > ptSeedThr &&
	    fabs((*p)->momentum().eta()) < etaSeedThr) {
        
	  // check if found is daughter of one already taken
	  bool isUsed = false;
	  
	  const GenParticle* mother = (*p)->production_vertex() ?
	    *((*p)->production_vertex()->particles_in_const_begin()) : 0;
	  const GenParticle* motherMother = (mother != 0  && mother->production_vertex()) ?
	    *(mother->production_vertex()->particles_in_const_begin()) : 0;
	  const GenParticle* motherMotherMother = (motherMother != 0 && motherMother->production_vertex()) ?
	    *(motherMother->production_vertex()->particles_in_const_begin()) : 0;

	  for(itPart = seeds.begin(); itPart != seeds.end(); itPart++) {
	    
	    if ((*itPart) == mother ||
		(*itPart) == motherMother ||
		(*itPart) == motherMotherMother) {
	      isUsed = true;
	      break;
	    }
	  }
	  
	  if (!isUsed) seeds.push_back(*p);
	}
      }  
  } 
   
  if (seeds.size() < 2) return accepted;

  for(HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p) {

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

  std::vector<int> nTracks;
  std::vector<TLorentzVector> candidate, candidateNarrow, candidateSeed;
  std::vector<const GenParticle*>::iterator itSeed;

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
    //bool isIsolated = true;

    if ( energy.Et() != 0. ) {
      if (fabs(energy.Eta()) < etaMaxCandidate) {

	temp2.SetXYZM(energy.Px(), energy.Py(), energy.Pz(), 0);        
	
	for(itStable = stable.begin(); itStable != stable.end(); ++itStable) {  
	  temp1.SetXYZM((*itStable)->momentum().px(), (*itStable)->momentum().py(), (*itStable)->momentum().pz(), 0);        
	  double DR = temp1.DeltaR(temp2);
	  if (DR < dRTkMax) counter++;        
	}

	if(acceptPrompts) {
	  bool isPrompt=false;
	  if((*itSeed)->status() == 1&&(*itSeed)->pdg_id() == 22) {
	    const GenParticle* mother = (*itSeed)->production_vertex() ?
	      *((*itSeed)->production_vertex()->particles_in_const_begin()) : 0;
	    if(mother) {
	      if(mother->pdg_id()>=-22&&mother->pdg_id()<=22) {
		const GenParticle* motherMother = (mother != 0  && mother->production_vertex()) ?
		  *(mother->production_vertex()->particles_in_const_begin()) : 0;
		if(motherMother) {
		  if(motherMother->pdg_id()>=-22&&motherMother->pdg_id()<=22) {
		    if((*itSeed)->momentum().perp()>promptPtThreshold) {
		      isPrompt=true;
		    }
		  }
		}
	      }
	    }
	  }
	  if(isPrompt) counter=0;
	}
	// check number of tracks
	//if (counter <= nTkConeMax) isIsolated = true;
      }
    }

    // check pt candidate, check nTrack, check eta
    //if (isIsolated) {
    candidate.push_back(energy);
    candidateSeed.push_back(tempseed);
    candidateNarrow.push_back(narrowCone);
    nTracks.push_back(counter);
    //++itSeed;
    //} 
  }

  if (candidate.size() <2) return accepted;

  TLorentzVector minv, minvNarrow;
  
  
  //bool filled=false;

  int i1, i2;
  for(unsigned int i=0; i<candidate.size()-1; ++i) {
    
    if (candidate[i].Energy() <1.) continue;
    if(nTracks[i]>nTkConeMax) continue;
    if (fabs(candidate[i].Eta()) > etaMaxCandidate) continue;
    
    for(unsigned int j=i+1; j<candidate.size(); ++j) { 
      if (candidate[j].Energy() <1.) continue;
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

      minv = candidate[i] + candidate[j];
      if (minv.M() < invMassWide) continue;
        
      minvNarrow = candidateNarrow[i] + candidateNarrow[j];
      if (minvNarrow.M() > invMassNarrow) continue;

      accepted = true;

      //if(!filled) {
      //hMassWide->Fill(minv.M());
      //hMassNarrow->Fill(minvNarrow.M());
      //hNTkSum->Fill(nTracks[i] + nTracks[j]);
      //hPtCandidate[0]->Fill(candidate[i1].Pt());
      //hPtCandidate[1]->Fill(candidate[i2].Pt());
      //hEtaCandidate[0]->Fill(candidate[i1].Eta());
      //hEtaCandidate[1]->Fill(candidate[i2].Eta());
      //hPidCandidate[0]->Fill(seeds[i1]->pdg_id());
      //hPidCandidate[1]->Fill(seeds[i2]->pdg_id());
      //hPtSeed[0]->Fill(candidateSeed[i1].Pt());
      //hPtSeed[1]->Fill(candidateSeed[i2].Pt());
      //hEtaSeed[0]->Fill(candidateSeed[i1].Eta());
      //hEtaSeed[1]->Fill(candidateSeed[i2].Eta());
      //hPidSeed[0]->Fill(seeds[i1]->pdg_id());
      //hPidSeed[1]->Fill(seeds[i2]->pdg_id());
      //hNTk[0]->Fill(nTracks[i1]);
      //hNTk[1]->Fill(nTracks[i2]);
      //filled=true;
      //}
    }
  }
  
  if (accepted) nSelectedEvents++;
  return accepted;
}

