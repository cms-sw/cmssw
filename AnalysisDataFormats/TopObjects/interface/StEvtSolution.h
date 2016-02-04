//
// $Id: StEvtSolution.h,v 1.13 2008/12/18 21:20:10 rwolf Exp $
//

#ifndef TopObjects_StEvtSolution_h
#define TopObjects_StEvtSolution_h

#include <vector>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/PatCandidates/interface/Particle.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

#include "AnalysisDataFormats/TopObjects/interface/StGenEvent.h"

class StEvtSolution {

  friend class StEvtSolutionMaker;
  friend class StKinFitter;
  
 public:

  StEvtSolution();
  virtual ~StEvtSolution();
  
  //-------------------------------------------
  // get calibrated base objects 
  //-------------------------------------------
  pat::Jet       getBottom()   const;
  pat::Jet       getLight()    const;
  pat::Muon      getMuon()     const { return *muon_; };
  pat::Electron  getElectron() const { return *electron_; };
  pat::MET       getNeutrino() const { return *neutrino_; };
  reco::Particle getLepW()     const;  
  reco::Particle getLept()     const;

  //-------------------------------------------
  // get the matched gen particles
  //-------------------------------------------
  const edm::RefProd<StGenEvent>& getGenEvent() const { return theGenEvt_; };
  const reco::GenParticle * getGenBottom()   const;
  //const reco::GenParticle * getGenLight() const; // not implemented yet
  const reco::GenParticle * getGenLepton()   const;
  const reco::GenParticle * getGenNeutrino() const;
  const reco::GenParticle * getGenLepW()     const;
  const reco::GenParticle * getGenLept()     const;

  //-------------------------------------------
  // get uncalibrated reco objects
  //-------------------------------------------
  pat::Jet       getRecBottom()   const { return this->getBottom().correctedJet("RAW"); };
  pat::Jet       getRecLight()    const { return this->getLight ().correctedJet("RAW"); };
  pat::Muon      getRecMuon()     const { return this->getMuon(); };     // redundant
  pat::Electron  getRecElectron() const { return this->getElectron(); }; // redundant
  pat::MET       getRecNeutrino() const { return this->getNeutrino(); }; // redundant
  reco::Particle getRecLepW()     const { return this->getLepW(); };     // redundant
  reco::Particle getRecLept()     const;

  //-------------------------------------------
  // get objects from kinematic fit
  //-------------------------------------------
  pat::Particle getFitBottom() const { return (fitBottom_.size()>0 ? fitBottom_.front() : pat::Particle()); };
  pat::Particle getFitLight() const { return (fitLight_.size()>0 ? fitLight_.front() : pat::Particle()); };
  pat::Particle getFitLepton() const { return (fitLepton_.size()>0 ? fitLepton_.front() : pat::Particle()); };
  pat::Particle getFitNeutrino() const { return (fitNeutrino_.size()>0 ? fitNeutrino_.front() : pat::Particle()); };
  reco::Particle getFitLepW() const;
  reco::Particle getFitLept() const;

  //-------------------------------------------
  // get info on the selected decay
  //-------------------------------------------
  std::string getDecay() const { return decay_; }

  //-------------------------------------------
  // get other event info
  //-------------------------------------------
  std::vector<double> getScanValues() const { return scanValues_; }
  double getChi2Prob()       const { return chi2Prob_; }
  double getPtrueCombExist() const { return pTrueCombExist_; }
  double getPtrueBJetSel()  const { return pTrueBJetSel_; }
  double getPtrueBhadrSel() const { return pTrueBhadrSel_; }
  double getPtrueJetComb() const { return pTrueJetComb_; }
  double getSignalPur()   const { return signalPur_; }
  double getSignalLRTot() const { return signalLRTot_; }
  double getSumDeltaRjp() const { return sumDeltaRjp_; }
  double getDeltaRB() const { return deltaRB_; }
  double getDeltaRL() const { return deltaRL_; }
  int  getChangeBL() const { return changeBL_; }
  bool getBestSol() const { return bestSol_; }
 
 protected:         

  //-------------------------------------------  
  // set the generated event
  //-------------------------------------------
  void setGenEvt(const edm::Handle<StGenEvent> &);

  //-------------------------------------------
  // set the basic objects
  //-------------------------------------------
  void setJetCorrectionScheme(int scheme) { jetCorrScheme_ = scheme;};
  void setBottom(const edm::Handle<std::vector<pat::Jet > >& jet, int i) 
  { bottom_ = edm::Ref<std::vector<pat::Jet> >(jet, i); };
  void setLight (const edm::Handle<std::vector<pat::Jet > >& jet, int i) 
  { light_ = edm::Ref<std::vector<pat::Jet> >(jet, i); };
  void setMuon  (const edm::Handle<std::vector<pat::Muon> >& muon, int i) 
  { muon_ = edm::Ref<std::vector<pat::Muon> >(muon, i); decay_ = "muon"; };
  void setElectron(const edm::Handle<std::vector<pat::Electron> >& elec, int i) 
  { electron_ = edm::Ref<std::vector<pat::Electron> >(elec, i); decay_ = "electron"; };
  void setNeutrino(const edm::Handle<std::vector<pat::MET> >& met, int i)
  { neutrino_ = edm::Ref<std::vector<pat::MET> >(met, i); };

  //-------------------------------------------
  // set the fitted objects 
  //-------------------------------------------
  void setFitBottom(const pat::Particle& part) { fitBottom_.clear(); fitBottom_.push_back(part); };
  void setFitLight (const pat::Particle& part) { fitLight_.clear(); fitLight_.push_back(part); };
  void setFitLepton(const pat::Particle& part) { fitLepton_.clear(); fitLepton_.push_back(part); };
  void setFitNeutrino(const pat::Particle& part) { fitNeutrino_.clear(); fitNeutrino_.push_back(part); };

  //-------------------------------------------
  // set other info on the event
  //-------------------------------------------
  void setChi2Prob(double prob){ chi2Prob_ = prob; };
  void setScanValues(const std::vector<double>&);
  void setPtrueCombExist(double pce){ pTrueCombExist_ = pce; };
  void setPtrueBJetSel (double pbs) { pTrueBJetSel_   = pbs; };
  void setPtrueBhadrSel(double pbh) { pTrueBhadrSel_  = pbh; };
  void setPtrueJetComb (double pt)  { pTrueJetComb_   = pt;  };
  void setSignalPurity (double pur) { signalPur_ = pur; };
  void setSignalLRTot(double lrt){ signalLRTot_ = lrt; };
  void setSumDeltaRjp(double sdr){ sumDeltaRjp_ = sdr; };
  void setDeltaRB(double adr) { deltaRB_ = adr; };
  void setDeltaRL(double adr) { deltaRL_ = adr; };
  void setChangeBL(int bl) { changeBL_ = bl; };
  void setBestSol(bool bs) { bestSol_  = bs; };
  
 private:

  //-------------------------------------------  
  // particle content
  //-------------------------------------------
  edm::RefProd<StGenEvent> theGenEvt_;
  edm::Ref<std::vector<pat::Jet> >  bottom_, light_;
  edm::Ref<std::vector<pat::Muon> > muon_;
  edm::Ref<std::vector<pat::Electron> > electron_;
  edm::Ref<std::vector<pat::MET> > neutrino_;
  std::vector<pat::Particle> fitBottom_, fitLight_, fitLepton_, fitNeutrino_;

  //-------------------------------------------
  // miscellaneous
  //-------------------------------------------
  std::string decay_;
  int jetCorrScheme_;
  double chi2Prob_;
  std::vector<double> scanValues_;
  double pTrueCombExist_, pTrueBJetSel_, pTrueBhadrSel_, pTrueJetComb_;
  double signalPur_, signalLRTot_;
  double sumDeltaRjp_, deltaRB_, deltaRL_;
  int changeBL_;
  bool bestSol_;
  //double jetMatchPur_;
};

#endif
