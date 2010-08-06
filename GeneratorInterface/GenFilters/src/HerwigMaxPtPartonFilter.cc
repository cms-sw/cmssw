/*
Author: Brian L. Dorney
Date: July 29th 2010
Version: 1.0
First Release In: CMSSW_3_8_X

Modified From: PythiaFilter.cc

Special Thanks to Filip Moortgat

PURPOSE: This Filter is designed to run on Herwig Monte Carlo Event Files
(Pythia status codes are assumed to NOT BE EMULATED!!!!)

For a description of Herwig Status Codes, See: 
http://webber.home.cern.ch/webber/hw65_manual.html#htoc96
(Section 8.3.1)

This Filter Finds the Parton (|pdg_id()|=1,2,3,4,5 or 21) in the final state
(before hadronization WITHIN THE EVENT occurs) with the highest Transverse 
Momentum (Pt), and compares this to the filtering range as defined by the user
in a Python Config File.

i.e. This Filter Checks:
minptcut <= HighestPartonPt < maxptcut

If this is true, the event is accepted.
*/

#include "GeneratorInterface/GenFilters/interface/HerwigMaxPtPartonFilter.h"
 
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <iostream>

using namespace edm;
using namespace std;


HerwigMaxPtPartonFilter::HerwigMaxPtPartonFilter(const edm::ParameterSet& iConfig) :
  label_(iConfig.getUntrackedParameter("moduleLabel",std::string("generator"))),
  minptcut(iConfig.getUntrackedParameter("MinPt", 0.)),
  maxptcut(iConfig.getUntrackedParameter("MaxPt", 10000.)),
  processID(iConfig.getUntrackedParameter("ProcessID", 0)){
    //now do what ever initialization is needed
 
}
 
 
HerwigMaxPtPartonFilter::~HerwigMaxPtPartonFilter(){
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
 
}
 
 
//
// member functions
//
 
// ------------ method called to produce the data  ------------
bool HerwigMaxPtPartonFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup){
  using namespace edm;
  
  accepted = false; //The Accept/Reject Variable
  isParton = false; //Keeps track of whether a particle is a parton as described above
  maxPartonPt=0.0; //Self Explanatory
  ChosenPartonId=0; ChosenPartonSt=0;
  pos1stCluster=0; //keeps track of the position of the first herwig cluster in the event
		 //This is when Hadronization within the event occurs.
  counter = 0; //keeps track of the particle index in the event
  
  Handle<HepMCProduct> evt;
  iEvent.getByLabel(label_, evt);
 
  HepMC::GenEvent * myGenEvent = new  HepMC::GenEvent(*(evt->GetEvent()));
     
  if(processID == 0 || processID == myGenEvent->signal_process_id()) { //processId if statement
   
    //Find the Position of the 1st Herwig Cluster
    for ( HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();p != myGenEvent->particles_end(); ++p ) {     
      pos1stCluster++;
      if(abs((*p)->pdg_id())==91){
	//pos1stCluster = (*p) - myGenEvent->particles_begin();
	break;
      }
    }
    
    //Loop through the all particles in the event
    for ( HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();p != myGenEvent->particles_end(); ++p ) {     
      //parton criterion
      if( abs((*p)->pdg_id())==1 ||
	  abs((*p)->pdg_id())==2 ||
	  abs((*p)->pdg_id())==3 ||
	  abs((*p)->pdg_id())==4 ||
	  abs((*p)->pdg_id())==5 ||
	  abs((*p)->pdg_id())==21){
	isParton=true;
      }//end if parton criterion
      
      //status & position criterion
      if(isParton && ((*p)->status()==158 || (*p)->status()==159) && counter<pos1stCluster){
	//Find Maximum Pt of Partons
	if((*p)->momentum().perp()>maxPartonPt){
	  maxPartonPt=(*p)->momentum().perp();//Record Max Parton Pt
	  //ChosenPartonId=(*p)->pdg_id(); ChosenPartonSt=(*p)->status(); //Debugging Purposes
          //cout<<"The New Max Pt is " << maxPartonPt << endl; //Debugging Purposes
	}
      } //end if status & position criterion
      
      counter++;
      isParton=false;
    } //end for loop
    
    //The Actual Filtering Process
    if(maxPartonPt>=minptcut && maxPartonPt<maxptcut){
      accepted=true; //Accept the Event
      //cout<<"Parton Used to filter was " << ChosenPartonId << " with status " << ChosenPartonSt << " and Pt " << maxPartonPt << endl;	/Debugging Purposes
    }//End Filtering
    
  }//end processId if statement
  
  
  else{ accepted = true; }
 
  delete myGenEvent; 
 
  if (accepted){return true; } 
  else {return false;}
}
