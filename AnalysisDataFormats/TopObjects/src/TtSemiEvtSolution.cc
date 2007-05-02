// -*- C++ -*-
//
// Package:     TtSemiEvtSolution
// Class  :     TtSemiEvtSolution
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Wed May 10 11:48:25 CEST 2006
// $Id: TtSemiEvtSolution.cc,v 1.6 2007/05/01 14:44:00 heyninck Exp $
//

// system include files

// user include files
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"



TtSemiEvtSolution::TtSemiEvtSolution()
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


TtSemiEvtSolution::~TtSemiEvtSolution()
{
}


void TtSemiEvtSolution::setHadp(TopJetObject j)      		{ hadp = j; }
void TtSemiEvtSolution::setHadq(TopJetObject j)      		{ hadq = j; }
void TtSemiEvtSolution::setHadb(TopJetObject j)      		{ hadb = j; }
void TtSemiEvtSolution::setLepb(TopJetObject j)      		{ lepb = j; }
void TtSemiEvtSolution::setMuon(TopMuonObject m)		{ muon = m; decay = "muon";}
void TtSemiEvtSolution::setElectron(TopElectronObject e)	{ electron = e;  decay = "electron";}
void TtSemiEvtSolution::setMET(TopMETObject   n)		{ met = n; }
void TtSemiEvtSolution::setChi2(double c)     			{ chi2 = c; }
void TtSemiEvtSolution::setPtrueCombExist(double pce)		{ ptrueCombExist= pce; }
void TtSemiEvtSolution::setPtrueBJetSel(double pbs)		{ ptrueBJetSel= pbs; }
void TtSemiEvtSolution::setPtrueBhadrSel(double pbh)		{ ptrueBhadrSel= pbh; }
void TtSemiEvtSolution::setPtrueJetComb(double pt)      	{ ptrueJetComb= pt; }
void TtSemiEvtSolution::setSignalPurity(double c)		{ signalPur = c; }
void TtSemiEvtSolution::setSignalLRtot(double c)		{ signalLRtot = c; }
void TtSemiEvtSolution::setScanValues(std::vector<double> v)    {
  for(unsigned int i=0; i<v.size(); i++) scanValues.push_back(v[i]);
}
void TtSemiEvtSolution::setGenEvt(vector<Candidate *> particles){
  genHadp = (Particle) (*(particles[0]));
  genHadq = (Particle) (*(particles[1]));
  genHadb = (Particle) (*(particles[2]));
  genLepb = (Particle) (*(particles[3]));
  genLepl = (Particle) (*(particles[4]));
  genLepn = (Particle) (*(particles[5]));
  genHadW = (Particle) (*(particles[6]));
  genLepW = (Particle) (*(particles[7]));
  genHadt = (Particle) (*(particles[8]));
  genLept = (Particle) (*(particles[9]));
}
void TtSemiEvtSolution::setSumDeltaRjp(double sdr)		{ sumDeltaRjp = sdr; }
void TtSemiEvtSolution::setDeltaRhadp(double adr)		{ deltaRhadp  = adr; }
void TtSemiEvtSolution::setDeltaRhadq(double adr)		{ deltaRhadq  = adr; }
void TtSemiEvtSolution::setDeltaRhadb(double adr)		{ deltaRhadb  = adr; }
void TtSemiEvtSolution::setDeltaRlepb(double adr)		{ deltaRlepb  = adr; }
void TtSemiEvtSolution::setChangeWQ(int wq)			{ changeWQ    = wq;  }
void TtSemiEvtSolution::setBestSol(bool bs)			{ bestSol     = bs;  }
      


// return functions for reconstructed fourvectors
jetType TtSemiEvtSolution::getRecHadp() const 	  { return this->getHadp().getRecJet(); }
jetType TtSemiEvtSolution::getRecHadq() const 	  { return this->getHadq().getRecJet(); }
jetType TtSemiEvtSolution::getRecHadb() const 	  { return this->getHadb().getRecJet(); }
jetType TtSemiEvtSolution::getRecLepb() const 	  { return this->getLepb().getRecJet(); }  
TopMET   TtSemiEvtSolution::getRecLepn() const 	  { return this->getMET().getRecMET();  }  
TopMuon  TtSemiEvtSolution::getRecLepm() const 	  { return this->getMuon().getRecMuon(); }
TopElectron TtSemiEvtSolution::getRecLepe() const { return this->getElectron().getRecElectron(); }
Particle TtSemiEvtSolution::getRecHadW() const 	  { return Particle(0,this->getRecHadp().p4() + this->getRecHadq().p4(),math::XYZPoint()); }
Particle TtSemiEvtSolution::getRecHadt() const    { return Particle(0,this->getRecHadp().p4() + this->getRecHadq().p4() + this->getRecHadb().p4(),math::XYZPoint()); }
Particle TtSemiEvtSolution::getRecLepW() const    { 
  Particle p;
  if (this->getDecay() == "muon")     p = Particle(0,this->getRecLepm().p4() + this->getRecLepn().p4(),math::XYZPoint());
  if (this->getDecay() == "electron") p = Particle(0,this->getRecLepe().p4() + this->getRecLepn().p4(),math::XYZPoint());
  return p;
}
Particle TtSemiEvtSolution::getRecLept() const    { 
  Particle p;
  if (this->getDecay() == "muon")     p = Particle(0,this->getRecLepm().p4() + this->getRecLepn().p4() + this->getRecLepb().p4(),math::XYZPoint());
  if (this->getDecay() == "electron") p = Particle(0,this->getRecLepe().p4() + this->getRecLepn().p4() + this->getRecLepb().p4(),math::XYZPoint());
  return p;
}



// return functions for calibrated fourvectors
TopJet TtSemiEvtSolution::getCalHadp() const 	 { return this->getHadp().getLCalJet(); }
TopJet TtSemiEvtSolution::getCalHadq() const 	 { return this->getHadq().getLCalJet(); }
TopJet TtSemiEvtSolution::getCalHadb() const 	 { return this->getHadb().getBCalJet(); }
TopJet TtSemiEvtSolution::getCalLepb() const 	 { return this->getLepb().getBCalJet(); }  
Particle TtSemiEvtSolution::getCalHadW() const   { return Particle(0,this->getCalHadp().p4() + this->getCalHadq().p4(),math::XYZPoint()); }
Particle TtSemiEvtSolution::getCalHadt() const   { return Particle(0,this->getCalHadp().p4() + this->getCalHadq().p4() + this->getCalHadb().p4(),math::XYZPoint()); }
Particle TtSemiEvtSolution::getCalLept() const   { 
  Particle p;
  if (this->getDecay() == "muon")     p = Particle(0,this->getRecLepm().p4() + this->getRecLepn().p4() + this->getCalLepb().p4(),math::XYZPoint());
  if (this->getDecay() == "electron") p = Particle(0,this->getRecLepe().p4() + this->getRecLepn().p4() + this->getCalLepb().p4(),math::XYZPoint());
  return p;
}



// return functions for fitted fourvectors
TopParticle TtSemiEvtSolution::getFitHadp() const { return this->getHadp().getFitJet(); }
TopParticle TtSemiEvtSolution::getFitHadq() const { return this->getHadq().getFitJet(); }
TopParticle TtSemiEvtSolution::getFitHadb() const { return this->getHadb().getFitJet(); }
TopParticle TtSemiEvtSolution::getFitLepb() const { return this->getLepb().getFitJet(); }  
TopParticle TtSemiEvtSolution::getFitLepn() const { return this->getMET().getFitMET();   } 
TopParticle TtSemiEvtSolution::getFitLepm() const { return this->getMuon().getFitMuon(); }
TopParticle TtSemiEvtSolution::getFitLepe() const { return this->getElectron().getFitElectron(); }
Particle   TtSemiEvtSolution::getFitHadW() const  { return Particle(0,this->getFitHadp().p4() + this->getFitHadq().p4(),math::XYZPoint()); }
Particle   TtSemiEvtSolution::getFitHadt() const  { return Particle(0,this->getFitHadp().p4() + this->getFitHadq().p4() + this->getFitHadb().p4(),math::XYZPoint()); }
Particle TtSemiEvtSolution::getFitLepW() const    { 
  Particle p;
  if (this->getDecay() == "muon")     p = Particle(0,this->getFitLepm().p4() + this->getFitLepn().p4(),math::XYZPoint());
  if (this->getDecay() == "electron") p = Particle(0,this->getFitLepe().p4() + this->getFitLepn().p4(),math::XYZPoint());
  return p;
}
Particle TtSemiEvtSolution::getFitLept() const   { 
  Particle p;
  if (this->getDecay() == "muon")     p = Particle(0,this->getFitLepm().p4() + this->getFitLepn().p4() + this->getFitLepb().p4(),math::XYZPoint());
  if (this->getDecay() == "electron") p = Particle(0,this->getFitLepe().p4() + this->getFitLepn().p4() + this->getFitLepb().p4(),math::XYZPoint());
  return p;
}
   
