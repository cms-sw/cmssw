//
// $Id: TtSemiEvtSolution.h,v 1.31 2013/04/19 22:13:23 wmtan Exp $
//

#ifndef TopObjects_TtSemiEvtSolution_h
#define TopObjects_TtSemiEvtSolution_h

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

#include "DataFormats/Candidate/interface/ShallowClonePtrCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"

// FIXME: make the decay member an enumerable
// FIXME: Can we generalize all the muon and electron to lepton?

class TtSemiEvtSolution {
  
  friend class TtSemiEvtSolutionMaker;
  friend class TtSemiLepKinFitter;
  friend class TtSemiLepHitFit;
  friend class TtSemiLRSignalSelObservables;
  friend class TtSemiLRSignalSelCalc;
  friend class TtSemiLRJetCombObservables;
  friend class TtSemiLRJetCombCalc;
  
 public:

  
  TtSemiEvtSolution();
  virtual ~TtSemiEvtSolution();
 
  //-------------------------------------------
  // get calibrated base objects 
  //------------------------------------------- 
  pat::Jet getHadb() const;
  pat::Jet getHadp() const;
  pat::Jet getHadq() const;
  pat::Jet getLepb() const;
  pat::Muon getMuon() const { return *muon_; };
  pat::Electron getElectron() const { return *electron_; };
  pat::MET getNeutrino() const { return *neutrino_; };

  //-------------------------------------------
  // get the matched gen particles
  //-------------------------------------------
  const edm::RefProd<TtGenEvent> & getGenEvent() const { return theGenEvt_; };
  const reco::GenParticle * getGenHadt() const { if (!theGenEvt_) return 0; else return this->getGenEvent()->hadronicDecayTop(); };
  const reco::GenParticle * getGenHadW() const { if (!theGenEvt_) return 0; else return this->getGenEvent()->hadronicDecayW(); };
  const reco::GenParticle * getGenHadb() const { if (!theGenEvt_) return 0; else return this->getGenEvent()->hadronicDecayB(); };
  const reco::GenParticle * getGenHadp() const { if (!theGenEvt_) return 0; else return this->getGenEvent()->hadronicDecayQuark(); };
  const reco::GenParticle * getGenHadq() const { if (!theGenEvt_) return 0; else return this->getGenEvent()->hadronicDecayQuarkBar(); };
  const reco::GenParticle * getGenLept() const { if (!theGenEvt_) return 0; else return this->getGenEvent()->leptonicDecayTop(); };
  const reco::GenParticle * getGenLepW() const { if (!theGenEvt_) return 0; else return this->getGenEvent()->leptonicDecayW(); };
  const reco::GenParticle * getGenLepb() const { if (!theGenEvt_) return 0; else return this->getGenEvent()->leptonicDecayB(); };
  const reco::GenParticle * getGenLepl() const { if (!theGenEvt_) return 0; else return this->getGenEvent()->singleLepton(); };
  const reco::GenParticle * getGenLepn() const { if (!theGenEvt_) return 0; else return this->getGenEvent()->singleNeutrino(); };

  //-------------------------------------------
  // get (un-)/calibrated reco objects
  //-------------------------------------------
  reco::Particle getRecHadt() const;
  reco::Particle getRecHadW() const;       
  pat::Jet getRecHadb() const { return this->getHadb().correctedJet("RAW"); };
  pat::Jet getRecHadp() const { return this->getHadp().correctedJet("RAW"); };
  pat::Jet getRecHadq() const { return this->getHadq().correctedJet("RAW"); };
  reco::Particle getRecLept() const;             
  reco::Particle getRecLepW() const;  
  pat::Jet getRecLepb() const { return this->getLepb().correctedJet("RAW"); }; 
  pat::Muon getRecLepm() const { return this->getMuon(); };
  pat::Electron getRecLepe() const { return this->getElectron(); };
  pat::MET getRecLepn() const { return this->getNeutrino(); };  
  // FIXME: Why these functions??? Not needed!
  // methods to get calibrated objects 
  reco::Particle getCalHadt() const;
  reco::Particle getCalHadW() const;
  pat::Jet getCalHadb() const { return this->getHadb(); };
  pat::Jet getCalHadp() const { return this->getHadp(); };
  pat::Jet getCalHadq() const { return this->getHadq(); };
  reco::Particle getCalLept() const;
  reco::Particle getCalLepW() const;
  pat::Jet getCalLepb() const { return this->getLepb(); };
  pat::Muon getCalLepm() const { return this->getMuon(); };
  pat::Electron getCalLepe() const { return this->getElectron(); };
  pat::MET getCalLepn() const { return this->getNeutrino(); };

  //-------------------------------------------
  // get objects from kinematic fit
  //-------------------------------------------  
  reco::Particle getFitHadt() const;
  reco::Particle getFitHadW() const;
  pat::Particle getFitHadb() const { return (fitHadb_.size()>0 ? fitHadb_.front() : pat::Particle()); };
  pat::Particle getFitHadp() const { return (fitHadp_.size()>0 ? fitHadp_.front() : pat::Particle()); };
  pat::Particle getFitHadq() const { return (fitHadq_.size()>0 ? fitHadq_.front() : pat::Particle()); };
  reco::Particle getFitLept() const;      
  reco::Particle getFitLepW() const;
  pat::Particle getFitLepb() const { return (fitLepb_.size()>0 ? fitLepb_.front() : pat::Particle()); };
  pat::Particle getFitLepl() const { return (fitLepl_.size()>0 ? fitLepl_.front() : pat::Particle()); }; 
  pat::Particle getFitLepn() const { return (fitLepn_.size()>0 ? fitLepn_.front() : pat::Particle()); };    

  //-------------------------------------------
  // get the selected semileptonic decay chain 
  //-------------------------------------------
  std::string getDecay() const { return decay_; }

  //-------------------------------------------
  // get info on the matching
  //-------------------------------------------
  double getMCBestSumAngles() const { return sumAnglejp_;};
  double getMCBestAngleHadp() const { return angleHadp_; };
  double getMCBestAngleHadq() const { return angleHadq_; };
  double getMCBestAngleHadb() const { return angleHadb_; };
  double getMCBestAngleLepb() const { return angleLepb_; };
  int getMCChangeWQ() const { return changeWQ_; };     

  //-------------------------------------------
  // get the selected kinfit parametrisations 
  // of each type of object 
  //-------------------------------------------
  int getJetParametrisation() const { return jetParam_; }
  int getLeptonParametrisation() const { return leptonParam_; }
  int getNeutrinoParametrisation() const { return neutrinoParam_; }

  //-------------------------------------------
  // get the prob of the chi2 value resulting 
  // from the kinematic fit
  //-------------------------------------------
  double getProbChi2() const { return probChi2_; }

  //-------------------------------------------
  // get info on the outcome of the signal 
  // selection LR
  //-------------------------------------------
  double getLRSignalEvtObsVal(unsigned int) const;
  double getLRSignalEvtLRval() const { return lrSignalEvtLRval_; }
  double getLRSignalEvtProb() const { return lrSignalEvtProb_; }

  //-------------------------------------------
  // get info on the outcome of the different 
  // jet combination methods
  //-------------------------------------------
  int getMCBestJetComb() const { return mcBestJetComb_; }
  int getSimpleBestJetComb() const { return simpleBestJetComb_; }
  int getLRBestJetComb() const { return lrBestJetComb_; }
  double getLRJetCombObsVal(unsigned int) const;
  double getLRJetCombLRval() const { return lrJetCombLRval_; }
  double getLRJetCombProb() const { return lrJetCombProb_; }



  //-------------------------------------------  
  // get the various event hypotheses
  //-------------------------------------------  
  const reco::CompositeCandidate & getRecoHyp() const { return recoHyp_; }
  const reco::CompositeCandidate & getFitHyp () const { return fitHyp_;  }
  const reco::CompositeCandidate & getMCHyp  () const { return mcHyp_;   }
  
 protected:         

  //-------------------------------------------  
  // set the generated event
  //-------------------------------------------
  void setGenEvt(const edm::Handle<TtGenEvent> & aGenEvt);

  //------------------------------------------- 
  // set the basic objects 
  //-------------------------------------------  
  void setJetCorrectionScheme(int scheme) { jetCorrScheme_ = scheme; };
  void setHadp(const edm::Handle<std::vector<pat::Jet> > & jet, int i)
  { hadp_ = edm::Ptr<pat::Jet>(jet, i); };
  void setHadq(const edm::Handle<std::vector<pat::Jet> > & jet, int i)
  { hadq_ = edm::Ptr<pat::Jet>(jet, i); };
  void setHadb(const edm::Handle<std::vector<pat::Jet> > & jet, int i)
  { hadb_ = edm::Ptr<pat::Jet>(jet, i); };
  void setLepb(const edm::Handle<std::vector<pat::Jet> > & jet, int i)
  { lepb_ = edm::Ptr<pat::Jet>(jet, i); };
  void setMuon(const edm::Handle<std::vector<pat::Muon> > & muon, int i)
  { muon_ = edm::Ptr<pat::Muon>(muon, i); decay_ = "muon"; };
  void setElectron(const edm::Handle<std::vector<pat::Electron> > & elec, int i)
  { electron_ = edm::Ptr<pat::Electron>(elec, i); decay_ = "electron"; };
  void setNeutrino(const edm::Handle<std::vector<pat::MET> > & met, int i)
  { neutrino_ = edm::Ptr<pat::MET>(met, i); };

  //-------------------------------------------  
  // set the fitted objects 
  //-------------------------------------------  
  void setFitHadb(const pat::Particle & aFitHadb) { fitHadb_.clear(); fitHadb_.push_back(aFitHadb); };
  void setFitHadp(const pat::Particle & aFitHadp) { fitHadp_.clear(); fitHadp_.push_back(aFitHadp); };
  void setFitHadq(const pat::Particle & aFitHadq) { fitHadq_.clear(); fitHadq_.push_back(aFitHadq); };
  void setFitLepb(const pat::Particle & aFitLepb) { fitLepb_.clear(); fitLepb_.push_back(aFitLepb); };
  void setFitLepl(const pat::Particle & aFitLepl) { fitLepl_.clear(); fitLepl_.push_back(aFitLepl); };
  void setFitLepn(const pat::Particle & aFitLepn) { fitLepn_.clear(); fitLepn_.push_back(aFitLepn); };

  //-------------------------------------------  
  // set the info on the matching
  //-------------------------------------------  
  void setMCBestSumAngles(double sdr) { sumAnglejp_= sdr; };
  void setMCBestAngleHadp(double adr) { angleHadp_ = adr; };
  void setMCBestAngleHadq(double adr) { angleHadq_ = adr; };
  void setMCBestAngleHadb(double adr) { angleHadb_ = adr; };
  void setMCBestAngleLepb(double adr) { angleLepb_ = adr; };
  void setMCChangeWQ(int wq) { changeWQ_ = wq; };

  //-------------------------------------------  
  // set the kinfit parametrisations of each 
  // type of object 
  //-------------------------------------------  
  void setJetParametrisation(int jp) { jetParam_ = jp; };
  void setLeptonParametrisation(int lp) { leptonParam_ = lp; };
  void setNeutrinoParametrisation(int mp) { neutrinoParam_ = mp; };

  //-------------------------------------------  
  // set the prob. of the chi2 value resulting 
  // from the kinematic fit 
  //-------------------------------------------  
  void setProbChi2(double c) { probChi2_ = c; };

  //-------------------------------------------  
  // set the outcome of the different jet 
  // combination methods
  //-------------------------------------------  
  void setMCBestJetComb(int mcbs) { mcBestJetComb_ = mcbs; };
  void setSimpleBestJetComb(int sbs) { simpleBestJetComb_ = sbs;  };
  void setLRBestJetComb(int lrbs) { lrBestJetComb_ = lrbs;  };
  void setLRJetCombObservables(const std::vector<std::pair<unsigned int, double> >& varval);
  void setLRJetCombLRval(double clr) {lrJetCombLRval_ = clr;};
  void setLRJetCombProb(double plr) {lrJetCombProb_ = plr;};

  //-------------------------------------------  
  // set the outcome of the signal selection LR
  //-------------------------------------------  
  void setLRSignalEvtObservables(const std::vector<std::pair<unsigned int, double> >& varval);
  void setLRSignalEvtLRval(double clr) {lrSignalEvtLRval_ = clr;};
  void setLRSignalEvtProb(double plr) {lrSignalEvtProb_ = plr;};



 private:

  //-------------------------------------------    
  // particle content
  //-------------------------------------------  
  edm::RefProd<TtGenEvent> theGenEvt_;
  edm::Ptr<pat::Jet> hadb_, hadp_, hadq_, lepb_;
  edm::Ptr<pat::Muon> muon_;
  edm::Ptr<pat::Electron> electron_;
  edm::Ptr<pat::MET> neutrino_;
  std::vector<pat::Particle> fitHadb_, fitHadp_, fitHadq_;
  std::vector<pat::Particle> fitLepb_, fitLepl_, fitLepn_;

  reco::CompositeCandidate mcHyp_;
  reco::CompositeCandidate recoHyp_;
  reco::CompositeCandidate fitHyp_;

  void setupHyp();

  std::string decay_;
  int jetCorrScheme_;
  double sumAnglejp_, angleHadp_, angleHadq_, angleHadb_, angleLepb_;
  int changeWQ_;
  int jetParam_, leptonParam_, neutrinoParam_;
  double probChi2_;
  int mcBestJetComb_, simpleBestJetComb_, lrBestJetComb_;
  double lrJetCombLRval_, lrJetCombProb_;
  double lrSignalEvtLRval_, lrSignalEvtProb_;
  std::vector<std::pair<unsigned int, double> > lrJetCombVarVal_;
  std::vector<std::pair<unsigned int, double> > lrSignalEvtVarVal_;  
};

#endif
