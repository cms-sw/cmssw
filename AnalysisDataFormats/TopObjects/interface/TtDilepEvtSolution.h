//
// $Id: TtDilepEvtSolution.h,v 1.23 2013/04/19 22:13:23 wmtan Exp $
//

#ifndef TopObjects_TtDilepEvtSolution_h
#define TopObjects_TtDilepEvtSolution_h

#include <vector>
#include <string>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

#include "DataFormats/PatCandidates/interface/Particle.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

class TtDilepEvtSolution {
  
  friend class TtFullLepKinSolver;
  friend class TtDilepEvtSolutionMaker;
  friend class TtDilepLRSignalSelObservables;
  friend class TtLRSignalSelCalc;
  
 public:
  
  TtDilepEvtSolution();
  virtual ~TtDilepEvtSolution();
  
  //-------------------------------------------
  // get calibrated base objects 
  //-------------------------------------------
  pat::Jet getJetB() const;
  pat::Jet getJetBbar() const;
  pat::Electron getElectronp() const { return *elecp_; };
  pat::Electron getElectronm() const { return *elecm_; };
  pat::Muon getMuonp() const { return *muonp_; };
  pat::Muon getMuonm() const { return *muonm_; };
  pat::Tau getTaup() const { return *taup_; };
  pat::Tau getTaum() const { return *taum_; };
  pat::MET getMET() const { return *met_; };

  //-------------------------------------------
  // get the matched gen particles
  //-------------------------------------------
  const edm::RefProd<TtGenEvent> & getGenEvent() const { return theGenEvt_; };
  const reco::GenParticle * getGenT() const { if (!theGenEvt_) return 0; else return theGenEvt_->top(); };
  const reco::GenParticle * getGenWp() const { if (!theGenEvt_) return 0; else return theGenEvt_->wPlus(); };
  const reco::GenParticle * getGenB() const { if (!theGenEvt_) return 0; else return theGenEvt_->b(); };
  const reco::GenParticle * getGenLepp() const { if (!theGenEvt_) return 0; else return theGenEvt_->leptonBar(); };
  const reco::GenParticle * getGenN() const { if (!theGenEvt_) return 0; else return theGenEvt_->neutrino(); };
  const reco::GenParticle * getGenTbar() const { if (!theGenEvt_) return 0; else return theGenEvt_->topBar(); };
  const reco::GenParticle * getGenWm() const { if (!theGenEvt_) return 0; else return theGenEvt_->wMinus(); };
  const reco::GenParticle * getGenBbar() const { if (!theGenEvt_) return 0; else return theGenEvt_->bBar(); };
  const reco::GenParticle * getGenLepm() const { if (!theGenEvt_) return 0; else return theGenEvt_->lepton(); };
  const reco::GenParticle * getGenNbar() const { if (!theGenEvt_) return 0; else return theGenEvt_->neutrinoBar(); };

  //-------------------------------------------
  // get (un-)/calibrated reco objects
  //-------------------------------------------
  pat::Jet      getRecJetB() const { return this->getJetB().correctedJet("RAW"); };
  pat::Jet      getCalJetB() const { return this->getJetB(); };
  pat::Jet      getRecJetBbar() const { return this->getJetBbar().correctedJet("RAW"); };
  pat::Jet      getCalJetBbar() const { return this->getJetBbar(); };

  //-------------------------------------------
  // get info on the W decays
  //-------------------------------------------
  std::string getWpDecay() const { return wpDecay_; }
  std::string getWmDecay() const { return wmDecay_; }

  //-------------------------------------------
  // miscellaneous
  //-------------------------------------------
  double getJetResidual()    const;
  double getLeptonResidual() const;
  double getFullResidual()   const { return getJetResidual()+getFullResidual(); }
  bool   getBestSol()      const { return bestSol_; }
  double getRecTopMass()   const {return topmass_; }
  double getRecWeightMax() const {return weightmax_; }
  
  //-------------------------------------------
  // returns the 4-vector of the positive 
  // lepton, with the charge and the pdgId
  //-------------------------------------------
  reco::Particle getLeptPos() const;

  //-------------------------------------------
  // returns the 4-vector of the negative 
  // lepton, with the charge and the pdgId
  //-------------------------------------------
  reco::Particle getLeptNeg() const;
  
  //-------------------------------------------
  // get info on the outcome of the signal 
  // selection LR
  //-------------------------------------------
  double getLRSignalEvtObsVal(unsigned int) const;
  double getLRSignalEvtLRval() const { return lrSignalEvtLRval_; }
  double getLRSignalEvtProb() const { return lrSignalEvtProb_; }
  
 protected:
  
  //-------------------------------------------
  // set the generated event
  //-------------------------------------------
  void setGenEvt(const edm::Handle<TtGenEvent>&);

  //-------------------------------------------
  // set the basic objects
  //-------------------------------------------
  void setJetCorrectionScheme(int jetCorrScheme) 
  { jetCorrScheme_ = jetCorrScheme; };
  void setB(const edm::Handle<std::vector<pat::Jet> >& jet, int i)
  { jetB_ = edm::Ref<std::vector<pat::Jet> >(jet, i); };
  void setBbar(const edm::Handle<std::vector<pat::Jet> >& jet, int i)
  { jetBbar_ = edm::Ref<std::vector<pat::Jet> >(jet, i); };
  void setMuonp(const edm::Handle<std::vector<pat::Muon> >& muon, int i)
  { muonp_ = edm::Ref<std::vector<pat::Muon> >(muon, i); wpDecay_ = "muon"; };
  void setMuonm(const edm::Handle<std::vector<pat::Muon> >& muon, int i)
  { muonm_ = edm::Ref<std::vector<pat::Muon> >(muon, i); wmDecay_ = "muon"; }
  void setTaup(const edm::Handle<std::vector<pat::Tau> >& tau, int i)
  { taup_ = edm::Ref<std::vector<pat::Tau> >(tau, i); wpDecay_ = "tau"; }
  void setTaum(const edm::Handle<std::vector<pat::Tau> >& tau, int i)
  { taum_ = edm::Ref<std::vector<pat::Tau> >(tau, i); wmDecay_ = "tau"; }
  void setElectronp(const edm::Handle<std::vector<pat::Electron> >& elec, int i)
  { elecp_ = edm::Ref<std::vector<pat::Electron> >(elec, i); wpDecay_ = "electron"; };
  void setElectronm(const edm::Handle<std::vector<pat::Electron> >& elec, int i)
  { elecm_ = edm::Ref<std::vector<pat::Electron> >(elec, i); wmDecay_ = "electron"; };
  void setMET(const edm::Handle<std::vector<pat::MET> >& met, int i)
  { met_ = edm::Ref<std::vector<pat::MET> >(met, i); };

  //-------------------------------------------
  // miscellaneous
  //-------------------------------------------
  void setBestSol(bool bs) { bestSol_ = bs; };
  void setRecTopMass(double mass) { topmass_ = mass; };
  void setRecWeightMax(double wgt) { weightmax_ = wgt; };
  
  //-------------------------------------------
  // set the outcome of the signal selection LR
  //-------------------------------------------
  void setLRSignalEvtObservables(const std::vector<std::pair<unsigned int, double> >&);
  void setLRSignalEvtLRval(double clr) {lrSignalEvtLRval_ = clr;};
  void setLRSignalEvtProb(double plr)  {lrSignalEvtProb_  = plr;};
  
 private:
  
  //-------------------------------------------
  // particle content
  //-------------------------------------------
  edm::RefProd<TtGenEvent>            theGenEvt_;
  edm::Ref<std::vector<pat::Electron> > elecp_, elecm_;
  edm::Ref<std::vector<pat::Muon> > muonp_, muonm_;
  edm::Ref<std::vector<pat::Tau> > taup_, taum_;
  edm::Ref<std::vector<pat::Jet> > jetB_, jetBbar_;
  edm::Ref<std::vector<pat::MET> > met_;

  //-------------------------------------------
  // miscellaneous
  //-------------------------------------------
  int jetCorrScheme_;
  std::string wpDecay_;
  std::string wmDecay_;      
  bool bestSol_;
  double topmass_;
  double weightmax_;
  
  double lrSignalEvtLRval_, lrSignalEvtProb_;
  std::vector<std::pair<unsigned int, double> > lrSignalEvtVarVal_;
};

#endif
