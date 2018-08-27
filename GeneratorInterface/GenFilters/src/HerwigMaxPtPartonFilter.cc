/*
Author: Brian L. Dorney
Date: July 29th 2010
Version: 2.2
First Release In: CMSSW_3_8_X

Modified From: PythiaFilter.cc

Special Thanks to Filip Moortgat

PURPOSE: This Filter is designed to run on Herwig Monte Carlo Event Files
(Pythia status codes are assumed to NOT BE EMULATED!!!!)

For a description of Herwig Status Codes, See: 
https://arxiv.org/abs/hep-ph/0011363
(Section 8.3.1)

This Filter Finds all final state quarks (pdg_id=1,2,3,4 or 5, status=158 or 159) with Pt>1 GeV/c
that occur before the first cluster (pdg_id=91) appears in the event cascade. This is done per event.

Then a histogram (which is RESET EACH EVENT) 2D histogram is formed, the Final State quarks
are then plotted in eta-phi space.  These histogram entries are weighted by the quark Pt.

This forms a 2D eta-phi "jet" topology for each event, and acts as a very rudimentary jet algorithm

The maximum bin entry (i.e. "jet") in this histogram is the highest Pt "Jet" in the event.

This is then used for filtering.

The size of each bin in this 2D histogram corresponds roughly to a cone radius of 0.5

i.e. This Filter Checks:
minptcut <= Highest Pt "Jet" < maxptcut

If this is true, the event is accepted.
*/

#include "GeneratorInterface/GenFilters/interface/HerwigMaxPtPartonFilter.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <cmath>

using namespace edm;
using namespace std;

HerwigMaxPtPartonFilter::HerwigMaxPtPartonFilter(const edm::ParameterSet& iConfig) :
  token_(consumes<edm::HepMCProduct>(iConfig.getUntrackedParameter("moduleLabel",edm::InputTag("generator","unsmeared")))),
  minptcut(iConfig.getUntrackedParameter("MinPt", 0.)),
  maxptcut(iConfig.getUntrackedParameter("MaxPt", 10000.)),
  processID(iConfig.getUntrackedParameter("ProcessID", 0)){
    //now do what ever initialization is needed

  
  hFSPartons_JS_PtWgting = new TH2D("hFSPartons_JS_PtWgting","#phi-#eta Space of FS Partons (p_{T} wgt'ing)",20,-5.205,5.205,32,-M_PI,M_PI);
  
}
 
 
HerwigMaxPtPartonFilter::~HerwigMaxPtPartonFilter(){
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

  if(hFSPartons_JS_PtWgting) delete hFSPartons_JS_PtWgting;
  
}
 
 
//
// member functions
//
 
// ------------ method called to produce the data  ------------
bool HerwigMaxPtPartonFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup){
  
  //Histogram, reset each event
  hFSPartons_JS_PtWgting->Reset();
  
  bool accepted = false; //The Accept/Reject Variable
  bool isFSQuark = false; //Keeps track of whether a particle is a Final State Quark
  double maxPartonPt=0.0; //Self Explanatory

  //int ChosenPartonId=0, ChosenPartonSt=0;

  int pos1stCluster=0; //keeps track of the position of the first herwig cluster in the event

  //This is when Hadronization within the event occurs.
  long counter = 0; //keeps track of the particle index in the event
  
  Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);
  
  const HepMC::GenEvent * myGenEvent = evt->GetEvent();
  
  
  if(processID == 0 || processID == myGenEvent->signal_process_id()) { //processId if statement
    
    //Find the Position of the 1st Herwig Cluster
    for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();p != myGenEvent->particles_end(); ++p ) {     
      if(abs((*p)->pdg_id())==91){
	break;
      }
      pos1stCluster++; //Starts at Zero, like the Collection
    }
    
    //Loop through the all particles in the event
    for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();p != myGenEvent->particles_end(); ++p ) {     
      //"Garbage" Cut, 1 GeV/c Pt Cut on All Particles Considered
      if((*p)->momentum().perp()>1.0){
	//Final State Quark criterion
	if(abs((*p)->pdg_id())==1 || abs((*p)->pdg_id())==2 || abs((*p)->pdg_id())==3 || abs((*p)->pdg_id())==4 || abs((*p)->pdg_id())==5){
	  if( counter<pos1stCluster && ((*p)->status()==158 || (*p)->status()==159) ){
	    isFSQuark=true;
	  }
	}//end if FS Quark criterion
      }//End "Garbage" Cut
      
      if(isFSQuark){
	hFSPartons_JS_PtWgting->Fill( (*p)->momentum().eta(), (*p)->momentum().phi(), (*p)->momentum().perp()); //weighted by Particle Pt
      }
      
      counter++;
      isFSQuark=false;
    } //end all particles loop
    
    maxPartonPt=hFSPartons_JS_PtWgting->GetMaximum();
    
    //The Actual Filtering Process
    if(maxPartonPt>=minptcut && maxPartonPt<maxptcut){
      accepted=true; //Accept the Event

    }//End Filtering
  }//end processId if statement
  
  else{ accepted = true; }
 
  if (accepted){return true; } 
  else {return false;}
}
