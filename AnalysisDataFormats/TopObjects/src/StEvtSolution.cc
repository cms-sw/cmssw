// -*- C++ -*-
//
// Package:     StEvtSolution
// Class  :     StEvtSolution
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Wed May 9 12:30:00 CEST 2007
// $Id: StEvtSolution.cc,v 1.4 2007/06/23 07:17:21 lowette Exp $
//

// system include files

// user include files
#include "AnalysisDataFormats/TopObjects/interface/StEvtSolution.h"



StEvtSolution::StEvtSolution()
{
  chi2 		= -999.;
  jetMatchPur  	= -999.;
  signalPur   	= -999.;
  sumDeltaRjp   = -999.;
  deltaRB   	= -999.;
  deltaRL   	= -999.;
  changeBL     	= -999;
  bestSol	= false;
}


StEvtSolution::~StEvtSolution()
{
}


void StEvtSolution::setBottom(TopJet j)      		{ bottom = j; }
void StEvtSolution::setLight(TopJet j)      		{ light = j; }
void StEvtSolution::setMuon(TopMuon m)		{ muon = m; decay = "muon";}
void StEvtSolution::setElectron(TopElectron e)	{ electron = e;  decay = "electron";}
void StEvtSolution::setMET(TopMET   n)		{ met = n; }
void StEvtSolution::setChi2(double c)     			{ chi2 = c; }
void StEvtSolution::setPtrueCombExist(double pce)		{ ptrueCombExist= pce; }
void StEvtSolution::setPtrueBJetSel(double pbs)		{ ptrueBJetSel= pbs; }
void StEvtSolution::setPtrueBhadrSel(double pbh)		{ ptrueBhadrSel= pbh; }
void StEvtSolution::setPtrueJetComb(double pt)      	{ ptrueJetComb= pt; }
void StEvtSolution::setSignalPurity(double c)		{ signalPur = c; }
void StEvtSolution::setSignalLRtot(double c)		{ signalLRtot = c; }
void StEvtSolution::setScanValues(std::vector<double> v)    {
  for(unsigned int i=0; i<v.size(); i++) scanValues.push_back(v[i]);
}
void StEvtSolution::setGenEvt(std::vector<reco::Candidate *> particles){
  genBottom = (reco::Particle) (*(particles[0]));
  genLight = (reco::Particle) (*(particles[1]));
  genLepl = (reco::Particle) (*(particles[2]));
  genLepn = (reco::Particle) (*(particles[3]));
  genLepW = (reco::Particle) (*(particles[4]));
  genLept = (reco::Particle) (*(particles[5]));
}
void StEvtSolution::setSumDeltaRjp(double sdr)		{ sumDeltaRjp = sdr; }
void StEvtSolution::setDeltaRB(double adr)		{ deltaRB  = adr; }
void StEvtSolution::setDeltaRL(double adr)		{ deltaRL  = adr; }
void StEvtSolution::setChangeBL(int bl)			{ changeBL    = bl;  }
void StEvtSolution::setBestSol(bool bs)			{ bestSol     = bs;  }
      


// return functions for reconstructed fourvectors
TopJetType StEvtSolution::getRecBottom() const 	  { return this->getBottom().getRecJet(); }
TopJetType StEvtSolution::getRecLight() const 	  { return this->getLight().getRecJet(); }
TopMET   StEvtSolution::getRecLepn() const 	  { return this->getMET();  }  
TopMuon  StEvtSolution::getRecLepm() const 	  { return this->getMuon(); }
TopElectron StEvtSolution::getRecLepe() const { return this->getElectron(); }
reco::Particle StEvtSolution::getRecLepW() const    { 
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0,this->getRecLepm().p4() + this->getRecLepn().p4(),math::XYZPoint());
  if (this->getDecay() == "electron") p = reco::Particle(0,this->getRecLepe().p4() + this->getRecLepn().p4(),math::XYZPoint());
  return p;
}
reco::Particle StEvtSolution::getRecLept() const    { 
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0,this->getRecLepm().p4() + this->getRecLepn().p4() + this->getRecBottom().p4(),math::XYZPoint());
  if (this->getDecay() == "electron") p = reco::Particle(0,this->getRecLepe().p4() + this->getRecLepn().p4() + this->getRecBottom().p4(),math::XYZPoint());
  return p;
}



// return functions for calibrated fourvectors
TopJet StEvtSolution::getCalLight() const 	 { return this->getLight(); }
TopJet StEvtSolution::getCalBottom() const 	 { return this->getBottom(); }
reco::Particle StEvtSolution::getCalLept() const   { 
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0,this->getRecLepm().p4() + this->getRecLepn().p4() + this->getCalBottom().p4(),math::XYZPoint());
  if (this->getDecay() == "electron") p = reco::Particle(0,this->getRecLepe().p4() + this->getRecLepn().p4() + this->getCalBottom().p4(),math::XYZPoint());
  return p;
}

// FIXME FIXME FIXME
// fit members must become part of the final state object

/*
// return functions for fitted fourvectors
TopParticle StEvtSolution::getFitBottom() const { return this->getBottom().getFitJet(); }
TopParticle StEvtSolution::getFitLight() const { return this->getLight().getFitJet(); }
TopParticle StEvtSolution::getFitLepn() const { return this->getMET().getFitMET();   } 
TopParticle StEvtSolution::getFitLepm() const { return this->getMuon().getFitLepton(); }
TopParticle StEvtSolution::getFitLepe() const { return this->getElectron().getFitLepton(); }
reco::Particle StEvtSolution::getFitLepW() const    { 
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0,this->getFitLepm().p4() + this->getFitLepn().p4(),math::XYZPoint());
  if (this->getDecay() == "electron") p = reco::Particle(0,this->getFitLepe().p4() + this->getFitLepn().p4(),math::XYZPoint());
  return p;
}
reco::Particle StEvtSolution::getFitLept() const   { 
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0,this->getFitLepm().p4() + this->getFitLepn().p4() + this->getFitBottom().p4(),math::XYZPoint());
  if (this->getDecay() == "electron") p = reco::Particle(0,this->getFitLepe().p4() + this->getFitLepn().p4() + this->getFitBottom().p4(),math::XYZPoint());
  return p;
}
*/   
