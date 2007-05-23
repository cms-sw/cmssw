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
// $Id: TtSemiEvtSolution.cc,v 1.5 2007/05/22 16:36:50 heyninck Exp $
//

// system include files

// user include files
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"



TtSemiEvtSolution::TtSemiEvtSolution()
{
  probChi2 		= -999.;
  sumDeltaRjp   	= -999.;
  deltaRhadp   		= -999.;
  deltaRhadq   		= -999.;
  deltaRhadb   		= -999.;
  deltaRlepb   		= -999.;
  changeWQ     		= -999;
  mcCorrJetComb		= -999;
  simpleCorrJetComb	= -999;
  lrCorrJetComb		= -999;
  lrCorrJetCombLRval	= -999.;
  lrCorrJetCombProb	= -999.;
  lrSignalEvtLRval	= -999.;
  lrSignalEvtProb	= -999.;
}


TtSemiEvtSolution::~TtSemiEvtSolution()
{
}


void TtSemiEvtSolution::setHadp(TopJet j)      			{ hadp = j; }
void TtSemiEvtSolution::setHadq(TopJet j)      			{ hadq = j; }
void TtSemiEvtSolution::setHadb(TopJet j)      			{ hadb = j; }
void TtSemiEvtSolution::setLepb(TopJet j)      			{ lepb = j; }
void TtSemiEvtSolution::setMuon(TopMuon m)			{ muon = m; decay = "muon";}
void TtSemiEvtSolution::setElectron(TopElectron e)		{ electron = e;  decay = "electron";}
void TtSemiEvtSolution::setMET(TopMET   n)			{ met = n; }
void TtSemiEvtSolution::setJetParametrisation(int jp) 		{ jetparam = jp; }
void TtSemiEvtSolution::setLeptonParametrisation(int lp) 	{ jetparam = lp; }
void TtSemiEvtSolution::setMETParametrisation(int mp) 		{ jetparam = mp; }
void TtSemiEvtSolution::setProbChi2(double c)     	        { probChi2 = c; }

void TtSemiEvtSolution::setGenEvt(std::vector<reco::Candidate *> particles){
  genHadp = (reco::Particle) (*(particles[0]));
  genHadq = (reco::Particle) (*(particles[1]));
  genHadb = (reco::Particle) (*(particles[2]));
  genLepb = (reco::Particle) (*(particles[3]));
  genLepl = (reco::Particle) (*(particles[4]));
  genLepn = (reco::Particle) (*(particles[5]));
  genHadW = (reco::Particle) (*(particles[6]));
  genLepW = (reco::Particle) (*(particles[7]));
  genHadt = (reco::Particle) (*(particles[8]));
  genLept = (reco::Particle) (*(particles[9]));
}
void TtSemiEvtSolution::setSumDeltaRjp(double sdr)		{ sumDeltaRjp   = sdr;  }
void TtSemiEvtSolution::setDeltaRhadp(double adr)		{ deltaRhadp    = adr;  }
void TtSemiEvtSolution::setDeltaRhadq(double adr)		{ deltaRhadq    = adr;  }
void TtSemiEvtSolution::setDeltaRhadb(double adr)		{ deltaRhadb    = adr;  }
void TtSemiEvtSolution::setDeltaRlepb(double adr)		{ deltaRlepb    = adr;  }
void TtSemiEvtSolution::setChangeWQ(int wq)			{ changeWQ      = wq;   }

void TtSemiEvtSolution::setMCCorrJetComb(int mcbs)		{ mcCorrJetComb = mcbs; }
void TtSemiEvtSolution::setSimpleCorrJetComb(int sbs)		{ simpleCorrJetComb = sbs;  }
void TtSemiEvtSolution::setLRCorrJetComb(int lrbs)		{ lrCorrJetComb = lrbs;  }
void TtSemiEvtSolution::setLRCorrJetCombVarVal(std::vector<std::pair<double, double> > varval) {
  for(size_t i = 0; i<varval.size(); i++) lrCorrJetCombVarVal.push_back(varval[i]);
}
void TtSemiEvtSolution::setLRCorrJetCombLRval(double clr) {lrCorrJetCombLRval = clr;}
void TtSemiEvtSolution::setLRCorrJetCombProb(double plr)  {lrCorrJetCombProb = plr;}
void TtSemiEvtSolution::setLRSignalEvtVarVal(std::vector<std::pair<double, double> > varval) {
  for(size_t i = 0; i<varval.size(); i++) lrSignalEvtVarVal.push_back(varval[i]);
}
void TtSemiEvtSolution::setLRSignalEvtLRval(double clr) {lrSignalEvtLRval = clr;}
void TtSemiEvtSolution::setLRSignalEvtProb(double plr)  {lrSignalEvtProb = plr;}
  


// return functions for reconstructed fourvectors
JetType TtSemiEvtSolution::getRecHadp() const 	  { return this->getHadp().getRecJet(); }
JetType TtSemiEvtSolution::getRecHadq() const 	  { return this->getHadq().getRecJet(); }
JetType TtSemiEvtSolution::getRecHadb() const 	  { return this->getHadb().getRecJet(); }
JetType TtSemiEvtSolution::getRecLepb() const 	  { return this->getLepb().getRecJet(); }  
TopMET   TtSemiEvtSolution::getRecLepn() const 	  { return this->getMET();  }  
TopMuon  TtSemiEvtSolution::getRecLepm() const 	  { return this->getMuon(); }
TopElectron TtSemiEvtSolution::getRecLepe() const { return this->getElectron(); }
reco::Particle TtSemiEvtSolution::getRecHadW() const 	  { return reco::Particle(0,this->getRecHadp().p4() + this->getRecHadq().p4(),math::XYZPoint()); }
reco::Particle TtSemiEvtSolution::getRecHadt() const    { return reco::Particle(0,this->getRecHadp().p4() + this->getRecHadq().p4() + this->getRecHadb().p4(),math::XYZPoint()); }
reco::Particle TtSemiEvtSolution::getRecLepW() const    { 
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0,this->getRecLepm().p4() + this->getRecLepn().p4(),math::XYZPoint());
  if (this->getDecay() == "electron") p = reco::Particle(0,this->getRecLepe().p4() + this->getRecLepn().p4(),math::XYZPoint());
  return p;
}
reco::Particle TtSemiEvtSolution::getRecLept() const    { 
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0,this->getRecLepm().p4() + this->getRecLepn().p4() + this->getRecLepb().p4(),math::XYZPoint());
  if (this->getDecay() == "electron") p = reco::Particle(0,this->getRecLepe().p4() + this->getRecLepn().p4() + this->getRecLepb().p4(),math::XYZPoint());
  return p;
}



// return functions for calibrated fourvectors
TopJet TtSemiEvtSolution::getCalHadp() const 	 { return this->getHadp(); }
TopJet TtSemiEvtSolution::getCalHadq() const 	 { return this->getHadq(); }
TopJet TtSemiEvtSolution::getCalHadb() const 	 { return this->getHadb(); }
TopJet TtSemiEvtSolution::getCalLepb() const 	 { return this->getLepb(); }  
reco::Particle TtSemiEvtSolution::getCalHadW() const   { return reco::Particle(0,this->getCalHadp().p4() + this->getCalHadq().p4(),math::XYZPoint()); }
reco::Particle TtSemiEvtSolution::getCalHadt() const   { return reco::Particle(0,this->getCalHadp().p4() + this->getCalHadq().p4() + this->getCalHadb().p4(),math::XYZPoint()); }
reco::Particle TtSemiEvtSolution::getCalLept() const   { 
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0,this->getRecLepm().p4() + this->getRecLepn().p4() + this->getCalLepb().p4(),math::XYZPoint());
  if (this->getDecay() == "electron") p = reco::Particle(0,this->getRecLepe().p4() + this->getRecLepn().p4() + this->getCalLepb().p4(),math::XYZPoint());
  return p;
}



// return functions for fitted fourvectors
TopParticle TtSemiEvtSolution::getFitHadp() const { return this->getHadp().getFitJet(); }
TopParticle TtSemiEvtSolution::getFitHadq() const { return this->getHadq().getFitJet(); }
TopParticle TtSemiEvtSolution::getFitHadb() const { return this->getHadb().getFitJet(); }
TopParticle TtSemiEvtSolution::getFitLepb() const { return this->getLepb().getFitJet(); }  
TopParticle TtSemiEvtSolution::getFitLepn() const { return this->getMET().getFitMET();   } 
TopParticle TtSemiEvtSolution::getFitLepm() const { return this->getMuon().getFitLepton(); }
TopParticle TtSemiEvtSolution::getFitLepe() const { return this->getElectron().getFitLepton(); }
reco::Particle   TtSemiEvtSolution::getFitHadW() const  { return reco::Particle(0,this->getFitHadp().p4() + this->getFitHadq().p4(),math::XYZPoint()); }
reco::Particle   TtSemiEvtSolution::getFitHadt() const  { return reco::Particle(0,this->getFitHadp().p4() + this->getFitHadq().p4() + this->getFitHadb().p4(),math::XYZPoint()); }
reco::Particle TtSemiEvtSolution::getFitLepW() const    { 
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0,this->getFitLepm().p4() + this->getFitLepn().p4(),math::XYZPoint());
  if (this->getDecay() == "electron") p = reco::Particle(0,this->getFitLepe().p4() + this->getFitLepn().p4(),math::XYZPoint());
  return p;
}
reco::Particle TtSemiEvtSolution::getFitLept() const   { 
  reco::Particle p;
  if (this->getDecay() == "muon")     p = reco::Particle(0,this->getFitLepm().p4() + this->getFitLepn().p4() + this->getFitLepb().p4(),math::XYZPoint());
  if (this->getDecay() == "electron") p = reco::Particle(0,this->getFitLepe().p4() + this->getFitLepn().p4() + this->getFitLepb().p4(),math::XYZPoint());
  return p;
}
   
