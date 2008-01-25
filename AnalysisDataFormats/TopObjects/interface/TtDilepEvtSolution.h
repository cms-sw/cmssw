//
// $Id: TtDilepEvtSolution.h,v 1.16 2008/01/17 10:07:35 speer Exp $
//

#ifndef TopObjects_TtDilepEvtSolution_h
#define TopObjects_TtDilepEvtSolution_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"
#include "AnalysisDataFormats/TopObjects/interface/TopElectron.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMuon.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMET.h"
#include "AnalysisDataFormats/TopObjects/interface/TopTau.h"

#include <vector>
#include <string>


class TtDilepEvtSolution {

  friend class TtDilepKinSolver;
  friend class TtDilepEvtSolutionMaker;
  friend class TtDilepLRSignalSelObservables;
  friend class TtLRSignalSelCalc;

  public:

    TtDilepEvtSolution();
    virtual ~TtDilepEvtSolution();

    // methods to et the original TopObjects
    TopJet      getJetB() const;
    TopJet      getJetBbar() const;
    TopElectron getElectronp() const;
    TopElectron getElectronm() const;
    TopMuon     getMuonp() const;
    TopMuon     getMuonm() const;
    TopTau      getTaup() const;
    TopTau      getTaum() const;
    TopMET      getMET() const;
    // methods to get the MC matched particles
    const edm::RefProd<TtGenEvent> & getGenEvent() const;
    const reco::GenParticle * getGenT() const;
    const reco::GenParticle * getGenWp() const;
    const reco::GenParticle * getGenB() const;
    const reco::GenParticle * getGenLepp() const;
    const reco::GenParticle * getGenN() const;
    const reco::GenParticle * getGenTbar() const;
    const reco::GenParticle * getGenWm() const;
    const reco::GenParticle * getGenBbar() const;
    const reco::GenParticle * getGenLepm() const;
    const reco::GenParticle * getGenNbar() const;
    // methods to explicitly get reconstructed and calibrated objects 
    TopJetType  getRecJetB() const;
    TopJet      getCalJetB() const;
    TopJetType  getRecJetBbar() const;
    TopJet      getCalJetBbar() const;
    // method to get info on the W decays
    std::string getWpDecay() const { return wpDecay_; }
    std::string getWmDecay() const { return wmDecay_; }
    // miscellaneous methods
    double getResidual()     const;
    bool   getBestSol()      const { return bestSol_; }
    double getRecTopMass()   const {return topmass_; }
    double getRecWeightMax() const {return weightmax_; }

  /**
   * Returns the 4-vector of the positive lepton, with the charge and the pdgId
   */
    reco::Particle getLeptPos() const;
  /**
   * Returns the 4-vector of the negative lepton, with the charge and the pdgId
   */
    reco::Particle getLeptNeg() const;

    // methods to get info on the outcome of the signal selection LR
    double                    getLRSignalEvtObsVal(unsigned int) const;
    double                    getLRSignalEvtLRval() const      { return lrSignalEvtLRval_; }
    double                    getLRSignalEvtProb() const       { return lrSignalEvtProb_; }

  protected:

    // method to set the generated event
    void setGenEvt(const edm::Handle<TtGenEvent> & aGenEvt);
    // methods to set the basic TopObjects
    void setJetCorrectionScheme(int jetCorrScheme);
    void setB(const edm::Handle<std::vector<TopJet> > & jh, int i);
    void setBbar(const edm::Handle<std::vector<TopJet> > & jh, int i);
    void setMuonp(const edm::Handle<std::vector<TopMuon> > & mh, int i);
    void setMuonm(const edm::Handle<std::vector<TopMuon> > & mh, int i);
    void setTaup(const edm::Handle<std::vector<TopTau> > & mh, int i);
    void setTaum(const edm::Handle<std::vector<TopTau> > & mh, int i);
    void setElectronp(const edm::Handle<std::vector<TopElectron> > & eh, int i);
    void setElectronm(const edm::Handle<std::vector<TopElectron> > & eh, int i);
    void setMET(const edm::Handle<std::vector<TopMET> > & nh, int i);
    // miscellaneous methods
    void setBestSol(bool bs);
    void setRecTopMass(double j);
    void setRecWeightMax(double j);

    // methods to set the outcome of the signal selection LR
    void                      setLRSignalEvtObservables(std::vector<std::pair<unsigned int, double> > varval);
    void                      setLRSignalEvtLRval(double clr);
    void                      setLRSignalEvtProb(double plr);

  private:

    // particle content
    edm::RefProd<TtGenEvent>            theGenEvt_;
    edm::Ref<std::vector<TopElectron> > elecp_, elecm_;
    edm::Ref<std::vector<TopMuon> >     muonp_, muonm_;
    edm::Ref<std::vector<TopTau> >      taup_, taum_;
    edm::Ref<std::vector<TopJet> >      jetB_, jetBbar_;
    edm::Ref<std::vector<TopMET> >      met_;
    // miscellaneous
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
