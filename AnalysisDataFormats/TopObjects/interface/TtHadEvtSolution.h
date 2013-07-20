#ifndef TopObjects_TtHadEvtSolution_h
#define TopObjects_TtHadEvtSolution_h
//
// $Id: TtHadEvtSolution.h,v 1.13 2013/04/19 22:13:23 wmtan Exp $
// adapted TtSemiEvtSolution.h,v 1.14 2007/07/06 03:07:47 lowette Exp 
// for fully hadronic channel

#include <vector>
#include <string>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

#include "DataFormats/PatCandidates/interface/Particle.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

class TtHadEvtSolution {
  
  friend class TtHadEvtSolutionMaker;
  friend class TtFullHadKinFitter;
  friend class TtHadLRJetCombObservables;
  friend class TtHadLRJetCombCalc;
  /*
    friend class TtHadLRSignalSelObservables;
    friend class TtHadLRSignalSelCalc;
  */
  
  public:
  
  TtHadEvtSolution();
  virtual ~TtHadEvtSolution();     
  
  //-------------------------------------------
  // get calibrated base objects 
  //-------------------------------------------
  pat::Jet getHadb() const;
  pat::Jet getHadp() const;
  pat::Jet getHadq() const;
  pat::Jet getHadbbar() const;
  pat::Jet getHadj() const;
  pat::Jet getHadk() const;
  
  //-------------------------------------------
  // get the matched gen particles
  //-------------------------------------------
  const edm::RefProd<TtGenEvent> & getGenEvent() const { return theGenEvt_; };
  const reco::GenParticle * getGenHadb() const { if (!theGenEvt_) return 0; else return theGenEvt_->b(); };
  const reco::GenParticle * getGenHadbbar() const { if (!theGenEvt_) return 0; else return theGenEvt_->bBar(); };
  const reco::GenParticle * getGenHadp() const { if (!theGenEvt_) return 0; else return theGenEvt_->daughterQuarkOfWPlus(); };
  const reco::GenParticle * getGenHadq() const { if (!theGenEvt_) return 0; else return theGenEvt_->daughterQuarkBarOfWPlus(); };
  const reco::GenParticle * getGenHadj() const { if (!theGenEvt_) return 0; else return theGenEvt_->daughterQuarkOfWMinus(); };
  const reco::GenParticle * getGenHadk() const { if (!theGenEvt_) return 0; else return theGenEvt_->daughterQuarkBarOfWMinus(); };
  
  //-------------------------------------------
  // get (un-)/calibrated reco objects
  //-------------------------------------------
  reco::Particle getRecHadt() const;
  reco::Particle getRecHadtbar() const;
  reco::Particle getRecHadW_plus() const;     
  reco::Particle getRecHadW_minus() const;       
  
  pat::Jet getRecHadb() const { return this->getHadb().correctedJet("RAW"); };
  pat::Jet getRecHadbbar() const { return this->getHadbbar().correctedJet("RAW"); };
  pat::Jet getRecHadp() const { return this->getHadp().correctedJet("RAW"); };
  pat::Jet getRecHadq() const { return this->getHadq().correctedJet("RAW"); };
  pat::Jet getRecHadj() const { return this->getHadj().correctedJet("RAW"); };
  pat::Jet getRecHadk() const { return this->getHadk().correctedJet("RAW"); };
  
  reco::Particle getCalHadt() const;
  reco::Particle getCalHadtbar() const;
  reco::Particle getCalHadW_plus() const;
  reco::Particle getCalHadW_minus() const;
  pat::Jet getCalHadb() const { return this->getHadb(); };
  pat::Jet getCalHadbbar() const { return this->getHadbbar(); };
  pat::Jet getCalHadp() const { return this->getHadp(); };
  pat::Jet getCalHadq() const { return this->getHadq(); };
  pat::Jet getCalHadj() const { return this->getHadj(); };
  pat::Jet getCalHadk() const { return this->getHadk(); };

  //-------------------------------------------
  // get objects from kinematic fit
  //-------------------------------------------  
  reco::Particle getFitHadt() const;
  reco::Particle getFitHadtbar() const;
  reco::Particle getFitHadW_plus() const;
  reco::Particle getFitHadW_minus() const;
  pat::Particle getFitHadb() const { return (fitHadb_.size()>0 ? fitHadb_.front() : pat::Particle()); };
  pat::Particle getFitHadbbar() const { return (fitHadbbar_.size()>0 ? fitHadbbar_.front() : pat::Particle()); };
  pat::Particle getFitHadp() const { return (fitHadp_.size()>0 ? fitHadp_.front() : pat::Particle()); };
  pat::Particle getFitHadq() const { return (fitHadq_.size()>0 ? fitHadq_.front() : pat::Particle()); };
  pat::Particle getFitHadj() const { return (fitHadj_.size()>0 ? fitHadj_.front() : pat::Particle()); };
  pat::Particle getFitHadk() const { return (fitHadk_.size()>0 ? fitHadk_.front() : pat::Particle()); };

  //-------------------------------------------  
  // get the selected hadronic decay chain 
  //-------------------------------------------
  std::string getDecay() const { return decay_; }

  //-------------------------------------------  
  // get info on the matching
  //-------------------------------------------  
  double getMCBestSumAngles() const { return sumAnglejp_; };
  double getMCBestAngleHadp() const { return angleHadp_; };
  double getMCBestAngleHadq() const { return angleHadq_; };
  double getMCBestAngleHadj() const { return angleHadj_; };
  double getMCBestAngleHadk() const { return angleHadk_; };
  double getMCBestAngleHadb() const { return angleHadb_; };
  double getMCBestAngleHadbbar() const { return angleHadbbar_; };
  int getMCChangeW1Q() const { return changeW1Q_; };     
  int getMCChangeW2Q() const { return changeW2Q_;}; 

  //-------------------------------------------  
  // get selected kinfit parametrisations of 
  //each type of object 
  //-------------------------------------------  
  int getJetParametrisation() const { return jetParam_; }

  //-------------------------------------------  
  // get the prob of the chi2 value resulting 
  // from the kinematic fit added chi2 for all 
  // fits
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
  //jet combination methods
  //-------------------------------------------  
  int getMCBestJetComb() const { return mcBestJetComb_; }
  int getSimpleBestJetComb() const { return simpleBestJetComb_; }
  int getLRBestJetComb() const { return lrBestJetComb_; }
  double getLRJetCombObsVal(unsigned int) const;
  double getLRJetCombLRval() const { return lrJetCombLRval_; }
  double getLRJetCombProb() const { return lrJetCombProb_; }
  
  //protected: seems to cause compile error, check!!!
  
  //-------------------------------------------  
  // set the generated event
  //-------------------------------------------  
  void setGenEvt(const edm::Handle<TtGenEvent> & aGenEvt);

  //-------------------------------------------
  // set the basic objects
  //-------------------------------------------  
  void setJetCorrectionScheme(int scheme) { jetCorrScheme_ = scheme; };
  void setHadp(const edm::Handle<std::vector<pat::Jet> > & jet, int i)
  { hadp_ = edm::Ref<std::vector<pat::Jet> >(jet, i); }
  void setHadq(const edm::Handle<std::vector<pat::Jet> > & jet, int i)
  { hadq_ = edm::Ref<std::vector<pat::Jet> >(jet, i); };
  void setHadj(const edm::Handle<std::vector<pat::Jet> > & jet, int i)
  { hadj_ = edm::Ref<std::vector<pat::Jet> >(jet, i); };
  void setHadk(const edm::Handle<std::vector<pat::Jet> > & jet, int i)
  { hadk_ = edm::Ref<std::vector<pat::Jet> >(jet, i); };
  void setHadb(const edm::Handle<std::vector<pat::Jet> > & jet, int i)
  { hadb_ = edm::Ref<std::vector<pat::Jet> >(jet, i); };
  void setHadbbar(const edm::Handle<std::vector<pat::Jet> > & jet, int i)
  { hadbbar_ = edm::Ref<std::vector<pat::Jet> >(jet, i); };

  //-------------------------------------------
  // set the fitted objects 
  //-------------------------------------------
  void setFitHadp(const pat::Particle & aFitHadp) { fitHadp_.clear(); fitHadp_.push_back(aFitHadp); };
  void setFitHadq(const pat::Particle & aFitHadq) { fitHadq_.clear(); fitHadq_.push_back(aFitHadq); };
  void setFitHadj(const pat::Particle & aFitHadj) { fitHadj_.clear(); fitHadj_.push_back(aFitHadj); };
  void setFitHadk(const pat::Particle & aFitHadk) { fitHadk_.clear(); fitHadk_.push_back(aFitHadk); };
  void setFitHadb(const pat::Particle & aFitHadb) { fitHadb_.clear(); fitHadb_.push_back(aFitHadb); };
  void setFitHadbbar(const pat::Particle & aFitHadbbar) { fitHadbbar_.clear(); fitHadbbar_.push_back(aFitHadbbar); };

  //-------------------------------------------
  // set matching info
  //-------------------------------------------
  void setMCBestSumAngles(double sdr) { sumAnglejp_ = sdr; };
  void setMCBestAngleHadp(double adr) { angleHadp_ = adr; };
  void setMCBestAngleHadq(double adr) { angleHadq_ = adr; };
  void setMCBestAngleHadj(double adr) { angleHadj_ = adr; };
  void setMCBestAngleHadk(double adr) { angleHadk_ = adr; };
  void setMCBestAngleHadb(double adr) { angleHadb_ = adr; };
  void setMCBestAngleHadbbar(double adr) { angleHadbbar_ = adr; };
  void setMCChangeW1Q(int w1q) { changeW1Q_ = w1q; };
  void setMCChangeW2Q(int w2q) { changeW2Q_ = w2q; };

  //-------------------------------------------
  // methods to set the kinfit parametrisations 
  //of each type of object 
  //-------------------------------------------
  void setJetParametrisation(int jp) { jetParam_ = jp; };

  //-------------------------------------------
  // method to set the prob. of the chi2 value 
  //resulting from the kinematic fit 
  //-------------------------------------------
  void setProbChi2(double c) { probChi2_ = c; };

  //-------------------------------------------
  // methods to set the outcome of the different 
  // jet combination methods
  //-------------------------------------------
  void setMCBestJetComb(int mcbs) { mcBestJetComb_ = mcbs; };
  void setSimpleBestJetComb(int sbs) { simpleBestJetComb_ = sbs; };
  void setLRBestJetComb(int lrbs) { lrBestJetComb_ = lrbs; };
  void setLRJetCombObservables(const std::vector<std::pair<unsigned int, double> >& varval);
  void setLRJetCombLRval(double clr) {lrJetCombLRval_ = clr;};
  void setLRJetCombProb(double plr) {lrJetCombProb_ = plr;};

  //-------------------------------------------
  // methods to set the outcome of the signal 
  // selection LR
  //-------------------------------------------
  void setLRSignalEvtObservables(const std::vector<std::pair<unsigned int, double> >& varval);
  void setLRSignalEvtLRval(double clr) {lrSignalEvtLRval_ = clr;};
  void setLRSignalEvtProb(double plr) {lrSignalEvtProb_ = plr;};
  
 private:

  //-------------------------------------------  
  // particle content
  //-------------------------------------------  
  edm::RefProd<TtGenEvent> theGenEvt_;
  edm::Ref<std::vector<pat::Jet> > hadb_, hadp_, hadq_, hadbbar_,hadj_, hadk_;
  std::vector<pat::Particle> fitHadb_, fitHadp_, fitHadq_, fitHadbbar_, fitHadj_, fitHadk_;
  
  std::string decay_;
  int jetCorrScheme_;
  double sumAnglejp_, angleHadp_, angleHadq_, angleHadb_, angleHadbbar_, angleHadj_ , angleHadk_;
  int changeW1Q_, changeW2Q_;
  int jetParam_;
  double probChi2_;
  int mcBestJetComb_, simpleBestJetComb_, lrBestJetComb_;
  double lrJetCombLRval_, lrJetCombProb_;
  double lrSignalEvtLRval_, lrSignalEvtProb_;
  std::vector<std::pair<unsigned int, double> > lrJetCombVarVal_;
  std::vector<std::pair<unsigned int, double> > lrSignalEvtVarVal_;
};

#endif
