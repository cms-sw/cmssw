/**
 * file ProtonTaggerFilter.cc
 *
 * Selection of very forward protons, generated and from pileup,
 * in clockwise and anti-clockwise beam directions.
 * Access to near-beam detector acceptances.
 *
 * Author: Dmitry Zaborov
 */

// Version: $Id: ProtonTaggerFilter.cc,v 1.3 2009/03/03 14:02:39 abdullin Exp $

#include "FastSimulation/ForwardDetectors/plugins/ProtonTaggerFilter.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Common/interface/Handle.h"


//#include "CLHEP/Random/RandGaussQ.h"


#include <iostream>
#include <list>


/** read (and verify) parameters */

ProtonTaggerFilter::ProtonTaggerFilter(edm::ParameterSet const & p)      
{
  std::cout << "ProtonTaggerFilter: Initializing ..." << std::endl;
  
  // ... get parameters

  beam1mode = p.getParameter<unsigned int>("beam1mode");
  beam2mode = p.getParameter<unsigned int>("beam2mode");

  beamCombiningMode = p.getParameter<unsigned int>("beamCombiningMode");
  
  switch (beam1mode)
  {
    case 0: std::cout << "Option chosen for beam 1: ingnore" << std::endl; break;
    case 1: std::cout << "Option chosen for beam 1: 420" << std::endl; break;
    case 2: std::cout << "Option chosen for beam 1: 220" << std::endl; break;
    case 3: std::cout << "Option chosen for beam 1: 420 and 220" << std::endl; break;
    case 4: std::cout << "Option chosen for beam 1: 420 or 220" << std::endl; break;
    default: throw cms::Exception("FastSimulation/ProtonTaggers") 
                           << "Error: beam1mode cannot be " << beam1mode;
  }
  
  switch (beam2mode)
  {
    case 0: std::cout << "Option chosen for beam 2: ingnore" << std::endl; break;
    case 1: std::cout << "Option chosen for beam 2: 420" << std::endl; break;
    case 2: std::cout << "Option chosen for beam 2: 220" << std::endl; break;
    case 3: std::cout << "Option chosen for beam 2: 420 and 220" << std::endl; break;
    case 4: std::cout << "Option chosen for beam 2: 420 or 220" << std::endl; break;
    default: throw cms::Exception("FastSimulation/ProtonTaggers")
                           << "Error: beam2mode cannot be " << beam2mode;
  }

  switch (beamCombiningMode)
  {
    case 1: std::cout << "Option chosen: one proton is sufficient" << std::endl; break;
    case 2: std::cout << "Option chosen: two protons should be tagged" << std::endl; break;
    case 3: std::cout << "Option chosen: two protons should be tagged as 220+220 or 420+420" << std::endl; break;
    case 4: std::cout << "Option chosen: two protons should be tagged as 220+420 or 420+220" << std::endl; break;
    default: throw cms::Exception("FastSimulation/ProtonTaggers") 
                             << "Error: beamCombiningMode cannot be " << beamCombiningMode;
  }

  if (((beam1mode != 4) || (beam2mode != 4)) && (beamCombiningMode > 2)) {
    std::cerr << "Warning: beamCombiningMode = " << beamCombiningMode 
              << " only makes sence with beam1mode = beam2mode = 4" << std::endl;
  }

  if (((beam1mode == 0) || (beam2mode == 0)) && (beamCombiningMode > 1)) {
    std::cerr << "Warning: You ask for 2 protons while one of the beams is set to ignore" << std::endl;
  }

  if ((beam1mode == 0) && (beam2mode == 0)) {
    std::cerr << "Warning: Both beams are set to ignore." << std::endl;
  }

  std::cout << "ProtonTaggerFilter: Initialized" << std::endl;
}


/** just empty */

ProtonTaggerFilter::~ProtonTaggerFilter() {;}


/** initialize detector acceptance table */

void ProtonTaggerFilter::beginJob()
{
  std::cout << "ProtonTaggerFilter: Getting ready ..." << std::endl;

  edm::FileInPath myDataFile("FastSimulation/ForwardDetectors/data/acceptance_420_220.root");
  std::string fullPath = myDataFile.fullPath();

  std::cout << "Opening " << fullPath << std::endl;
  TFile f(fullPath.c_str());

  if (f.Get("description") != NULL)
      std::cout << "Description found: " << f.Get("description")->GetTitle() << std::endl;
  
  std::cout << "Reading acceptance tables @#@#%@$%@$#%@%" << std::endl;
  
  helper420beam1.Init(f, "a420");
  helper420beam2.Init(f, "a420_b2");

  helper220beam1.Init(f, "a220");
  helper220beam2.Init(f, "a220_b2");

  helper420a220beam1.Init(f, "a420a220");
  helper420a220beam2.Init(f, "a420a220_b2");

  f.Close();

  std::cout << "ProtonTaggerFilter: Ready" << std::endl;
}


/** nothing to be done here */

void ProtonTaggerFilter::endJob() {;}


/** Compute the detector acceptances and decide whether to filter the event */

bool ProtonTaggerFilter::filter(edm::Event & iEvent, const edm::EventSetup & es)
{
  // ... get generated event

  edm::Handle<edm::HepMCProduct> evtSource;
  iEvent.getByLabel("generatorSmeared",evtSource);
  const HepMC::GenEvent* genEvent = evtSource->GetEvent();

  //std::cout << "event contains " << genEvent->particles_size() << " particles " << std::endl;
  if (genEvent->particles_empty()) {
    std::cout << "empty source event" << std::endl;
    return false;
  }

  // ... get pileup event

  edm::Handle<edm::HepMCProduct> pileUpSource;
  const HepMC::GenEvent* pileUpEvent = 0;
  bool isPileUp = true;
  
  bool isProduct = iEvent.getByLabel("famosPileUp", "PileUpEvents", pileUpSource);

  if (isProduct) {
    pileUpEvent = pileUpSource->GetEvent();
    //std::cout << "got pileup" << std::endl;
  } else {
    isPileUp = false;
    //std::cout << "no pileup in the event" << std::endl;
  }

  //if (isPileUp) std::cout << "event contains " << pileUpEvent->particles_size() << " pileup particles " << std::endl;  

  // ... some constants

  const double mp = 0.938272029; // just a proton mass
  const double Ebeam = 7000.0;   // beam energy - would be better to read from parameters
  const float  pzCut = 2500;     // ignore particles with less than |Pz| < pzCut  
  const float  acceptThreshold = 0.5; // proton will be "accepted" if the value of acceptance is above this threshold

  // ... loop over particles, find the most energetic proton in either direction

  std::list<HepMC::GenParticle*> veryForwardParicles;
  
  for ( HepMC::GenEvent::particle_const_iterator piter = genEvent->particles_begin();
          piter != genEvent->particles_end();
          ++piter )
  {
    
    HepMC::GenParticle* p = *piter;

    float pz = p->momentum().pz();
    if (((pz > pzCut) || (pz < -pzCut)) && ((p->status() == 0) || (p->status() == 1))) {
      veryForwardParicles.push_back(p);
      //std::cout << "pdgid: " << p->pdg_id() << " status: " << p->status() << std::endl;
    }
  }

  //std::cout << "# generated forward particles  : " << veryForwardParicles.size() << std::endl;

  if ( isPileUp ) {

    //std::cout << "Adding pileup " << std::endl;

    for ( HepMC::GenEvent::particle_const_iterator piter = pileUpEvent->particles_begin();
  	    piter != pileUpEvent->particles_end();
  	    ++piter )
    {
      
      HepMC::GenParticle* p = *piter;

      float pz = p->momentum().pz();
      if (((pz > pzCut) || (pz < -pzCut)) && ((p->status() == 0) || (p->status() == 1))) {
        veryForwardParicles.push_back(p);
        //std::cout << "pdgid: " << p->pdg_id() << " status: " << p->status() << std::endl;
      }
    }
  }

  //std::cout << "# forward particles to be tried: " << veryForwardParicles.size() << std::endl;


  // ... return false if no forward protons found

  if (veryForwardParicles.empty()) return false;


  // ... set all acceptances to zero

  float acc420b1, acc220b1, acc420and220b1, acc420or220b1; // beam 1 (clockwise)
  float acc420b2, acc220b2, acc420and220b2, acc420or220b2; // beam 2 (anti-clockwise)

  acc420b1 = acc220b1 = acc420and220b1 = acc420or220b1 = 0;
  acc420b2 = acc220b2 = acc420and220b2 = acc420or220b2 = 0;

  int nP1at220m = 0;
  int nP1at420m = 0;

  int nP2at220m = 0;
  int nP2at420m = 0;

  // ... loop over (pre-selected) forward particles
  
  for (std::list<HepMC::GenParticle*>::const_iterator part = veryForwardParicles.begin();
       part != veryForwardParicles.end();
       part++)
  {

    HepMC::GenParticle* p = *part;

    float pz  = p->momentum().pz();
    float pt  = p->momentum().perp();
    float phi = p->momentum().phi();;

    if ((pz > Ebeam) || (pz < -Ebeam)) continue;
    
    // ... compute kinimatical variable

    float xi  = 1.0;	// fractional momentum loss
    if (pz>0) xi -= pz/Ebeam;
    else xi += pz/Ebeam;

    double t   = (-pt*pt - mp*mp*xi*xi) / (1-xi); // "t"

    //std::cout << " pdg_id: "  << p->pdg_id() << " eta: " << p->momentum().eta() << " e: " 
    //          <<  p->momentum().e() << std::endl;
    //std::cout << "pz: " << pz << " pt: " <<  pt << " xi: " << xi
    //          << " t: " << t << " phi: " << phi << std::endl;

    if (xi<0.0) xi=-10.0;
    if (xi>1.0) xi=10.0;

    //float rnd1 = RandGauss::shoot(0.,1.);
    //float rnd2 = RandGauss::shoot(0.,1.);

    // ... get acceptance from tables: beam 1 (if it is not ignored)

    if ((pz > 0) && (beam1mode != 0)) {    

      acc420b1       = helper420beam1.GetAcceptance(t, xi, phi);
      acc220b1       = helper220beam1.GetAcceptance(t, xi, phi);
      acc420and220b1 = helper420a220beam1.GetAcceptance(t, xi, phi);

      acc420or220b1  = acc420b1 + acc220b1 - acc420and220b1;

      //std::cout << "+acc420b1: " << acc420b1 << " acc220b1: " << acc220b1 << " acc420and220b1: " << acc420and220b1 << " acc420or220b1: " << acc420or220b1  << std::endl;

      bool res420and220 = (acc420and220b1 > acceptThreshold);
      bool res420       = (acc420b1       > acceptThreshold);
      bool res220       = (acc220b1       > acceptThreshold);

      if (res420and220) {nP1at220m++; nP1at420m++;}
      else if (res420) nP1at420m++;
      else if (res220) nP1at220m++;
  
      if ((p->pdg_id() != 2212) && (res220 || res420 || res420and220)) {
        std::cout << " !!! P got proton 1 at 420 m: pz = " << pz << std::endl;
        if (res220) std::cout << "got a particle with pid" << p->pdg_id() << " at 220 m along beam 1, pz = " << pz << std::endl;
        if (res420) std::cout << "got a particle with pid" << p->pdg_id() << " at 420 m along beam 1, pz = " << pz << std::endl;
        if (res420and220) std::cout << "got a particle with pid" << p->pdg_id() << " at 220 m & 420 m  along beam 1, pz = " << pz << std::endl;
      }
    }
    
    // ... the same for beam 2

    if ((pz < 0) && (beam2mode != 0)) {

      acc420b2       = helper420beam2.GetAcceptance(t, xi, phi);
      acc220b2       = helper220beam2.GetAcceptance(t, xi, phi);
      acc420and220b2 = helper420a220beam2.GetAcceptance(t, xi, phi);
    
      acc420or220b2  = acc420b2 + acc220b2 - acc420and220b2;

      //std::cout << "+acc420b2: " << acc420b2 << " acc220b2: " << acc220b2 << " acc420and220b2: " << acc420and220b2 << " acc420or220b2: " << acc420or220b2 << std::endl;

      bool res420and220 = (acc420and220b2 > acceptThreshold);
      bool res420       = (acc420b2       > acceptThreshold);
      bool res220       = (acc220b2       > acceptThreshold);

      if (res420and220) {nP2at220m++; nP2at420m++;}
      else if (res420) nP2at420m++;
      else if (res220) nP2at220m++;
  
      if ((p->pdg_id() != 2212) && (res220 || res420 || res420and220)) {
        std::cout << " !!! P got proton 1 at 420 m: pz = " << pz << std::endl;
        if (res220) std::cout << "got a particle with pid" << p->pdg_id() << " at 220 m along beam 2, pz = " << pz << std::endl;
        if (res420) std::cout << "got a particle with pid" << p->pdg_id() << " at 420 m along beam 2, pz = " << pz << std::endl;
        if (res420and220) std::cout << "got a particle with pid" << p->pdg_id() << " at 220 m & 420 m along beam 2, pz = " << pz << std::endl;
      }
    }
  }

  // ... boolean result for each detector
  
  bool p1at220m = (nP1at220m>0) ? true : false;
  bool p1at420m = (nP1at420m>0) ? true : false;
  bool p2at220m = (nP2at220m>0) ? true : false;
  bool p2at420m = (nP2at420m>0) ? true : false;

  if ((nP1at220m > 1) && (beam1mode!=1)) std::cout << " !!! " << nP1at220m << " proton(s) from beam 1 at 220 m" << std::endl;
  if ((nP1at420m > 1) && (beam1mode!=2)) std::cout << " !!! " << nP1at420m << " proton(s) from beam 1 at 420 m" << std::endl;
  if ((nP2at220m > 1) && (beam2mode!=1)) std::cout << " !!! " << nP2at220m << " proton(s) from beam 2 at 220 m" << std::endl;
  if ((nP2at420m > 1) && (beam2mode!=2)) std::cout << " !!! " << nP2at420m << " proton(s) from beam 2 at 420 m" << std::endl;


  // ... make a decision based on requested filter configuration
    
  bool p1accepted = false;
  bool p2accepted = false;
  
  if ((beam1mode==1) &&  p1at420m ) p1accepted = true;
  if ((beam1mode==2) &&  p1at220m ) p1accepted = true;
  if ((beam1mode==3) &&  p1at220m && p1at420m) p1accepted = true;
  if ((beam1mode==4) && (p1at220m || p1at420m)) p1accepted = true;
  
  if ((beam2mode==1) &&  p2at420m ) p2accepted = true;
  if ((beam2mode==2) &&  p2at220m ) p2accepted = true;
  if ((beam2mode==3) &&  p2at220m && p2at420m ) p2accepted = true;
  if ((beam2mode==4) && (p2at220m || p2at420m)) p2accepted = true;

  //if (p1accepted) std::cout << "proton 1 accepted" << std::endl;
  //if (p2accepted) std::cout << "proton 2 accepted" << std::endl;

  switch (beamCombiningMode) {

    case 1:  // ... either of two protons
     if (p1accepted || p2accepted) return true; else return false;

    case 2:  // ... both protons
     if (p1accepted && p2accepted) return true; else return false;

    case 3:  // ... 220+220 or 420+420
     if ((p1at220m && p2at220m) || (p1at420m && p2at420m)) return true; else return false;

    case 4:  // ... 220+420 or 420+220
     if ((p1at220m && p2at420m) || (p1at420m && p2at220m)) return true; else return false;

  }
  
  return false;
}

// ... define CMSSW module

DEFINE_FWK_MODULE(ProtonTaggerFilter);
