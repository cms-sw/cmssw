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
// $Id: StEvtSolution.cc,v 0.0 2007/05/09 12:30:00 giamman Exp $
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
  deltaRhadp   	= -999.;
  deltaRhadq   	= -999.;
  deltaRhadb   	= -999.;
  deltaRlepb   	= -999.;
  changeWQ     	= -999;
  bestSol	= false;
}


StEvtSolution::~StEvtSolution()
{
}


void StEvtSolution::setBottom(TopJetObject j)      		{ bottom = j; }
void StEvtSolution::setLight(TopJetObject j)      		{ light = j; }
void StEvtSolution::setMuon(TopMuonObject m)		{ muon = m; decay = "muon";}
void StEvtSolution::setElectron(TopElectronObject e)	{ electron = e;  decay = "electron";}
void StEvtSolution::setMET(TopMETObject   n)		{ met = n; }
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
void StEvtSolution::setGenEvt(vector<Candidate *> particles){
  genBottom = (Particle) (*(particles[0]));
  genLight = (Particle) (*(particles[1]));
  genLepl = (Particle) (*(particles[2]));
  genLepn = (Particle) (*(particles[3]));
  genLepW = (Particle) (*(particles[4]));
  genLept = (Particle) (*(particles[5]));
}
void StEvtSolution::setSumDeltaRjp(double sdr)		{ sumDeltaRjp = sdr; }
void StEvtSolution::setDeltaRhadp(double adr)		{ deltaRhadp  = adr; }
void StEvtSolution::setDeltaRhadq(double adr)		{ deltaRhadq  = adr; }
void StEvtSolution::setDeltaRhadb(double adr)		{ deltaRhadb  = adr; }
void StEvtSolution::setDeltaRlepb(double adr)		{ deltaRlepb  = adr; }
void StEvtSolution::setChangeWQ(int wq)			{ changeWQ    = wq;  }
void StEvtSolution::setBestSol(bool bs)			{ bestSol     = bs;  }
      


// return functions for reconstructed fourvectors
JetType StEvtSolution::getRecBottom() const 	  { return this->getBottom().getRecJet(); }
JetType StEvtSolution::getRecLight() const 	  { return this->getLight().getRecJet(); }
TopMET   StEvtSolution::getRecLepn() const 	  { return this->getMET().getRecMET();  }  
TopMuon  StEvtSolution::getRecLepm() const 	  { return this->getMuon().getRecMuon(); }
TopElectron StEvtSolution::getRecLepe() const { return this->getElectron().getRecElectron(); }
Particle StEvtSolution::getRecLepW() const    { 
  Particle p;
  if (this->getDecay() == "muon")     p = Particle(0,this->getRecLepm().p4() + this->getRecLepn().p4(),math::XYZPoint());
  if (this->getDecay() == "electron") p = Particle(0,this->getRecLepe().p4() + this->getRecLepn().p4(),math::XYZPoint());
  return p;
}
Particle StEvtSolution::getRecLept() const    { 
  Particle p;
  if (this->getDecay() == "muon")     p = Particle(0,this->getRecLepm().p4() + this->getRecLepn().p4() + this->getRecBottom().p4(),math::XYZPoint());
  if (this->getDecay() == "electron") p = Particle(0,this->getRecLepe().p4() + this->getRecLepn().p4() + this->getRecBottom().p4(),math::XYZPoint());
  return p;
}



// return functions for calibrated fourvectors
TopJet StEvtSolution::getCalLight() const 	 { return this->getLight().getLCalJet(); }
TopJet StEvtSolution::getCalBottom() const 	 { return this->getBottom().getBCalJet(); }
Particle StEvtSolution::getCalLept() const   { 
  Particle p;
  if (this->getDecay() == "muon")     p = Particle(0,this->getRecLepm().p4() + this->getRecLepn().p4() + this->getCalBottom().p4(),math::XYZPoint());
  if (this->getDecay() == "electron") p = Particle(0,this->getRecLepe().p4() + this->getRecLepn().p4() + this->getCalBottom().p4(),math::XYZPoint());
  return p;
}



// return functions for fitted fourvectors
TopParticle StEvtSolution::getFitBottom() const { return this->getBottom().getFitJet(); }
TopParticle StEvtSolution::getFitLight() const { return this->getLight().getFitJet(); }
TopParticle StEvtSolution::getFitLepn() const { return this->getMET().getFitMET();   } 
TopParticle StEvtSolution::getFitLepm() const { return this->getMuon().getFitMuon(); }
TopParticle StEvtSolution::getFitLepe() const { return this->getElectron().getFitElectron(); }
Particle StEvtSolution::getFitLepW() const    { 
  Particle p;
  if (this->getDecay() == "muon")     p = Particle(0,this->getFitLepm().p4() + this->getFitLepn().p4(),math::XYZPoint());
  if (this->getDecay() == "electron") p = Particle(0,this->getFitLepe().p4() + this->getFitLepn().p4(),math::XYZPoint());
  return p;
}
Particle StEvtSolution::getFitLept() const   { 
  Particle p;
  if (this->getDecay() == "muon")     p = Particle(0,this->getFitLepm().p4() + this->getFitLepn().p4() + this->getFitBottom().p4(),math::XYZPoint());
  if (this->getDecay() == "electron") p = Particle(0,this->getFitLepe().p4() + this->getFitLepn().p4() + this->getFitBottom().p4(),math::XYZPoint());
  return p;
}
   
