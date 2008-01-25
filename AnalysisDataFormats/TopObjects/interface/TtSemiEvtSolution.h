//
// $Id: TtSemiEvtSolution.h,v 1.22 2007/11/24 11:03:15 lowette Exp $
//

#ifndef TopObjects_TtSemiEvtSolution_h
#define TopObjects_TtSemiEvtSolution_h

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

#include <vector>
#include <string>


// FIXME: make the decay member an enumerable
// FIXME: Can we generalize all the muon and electron to lepton?

class TtSemiEvtSolution {

  friend class TtSemiEvtSolutionMaker;
  friend class TtSemiKinFitter;
  friend class TtSemiLRSignalSelObservables;
  friend class TtSemiLRSignalSelCalc;
  friend class TtSemiLRJetCombObservables;
  friend class TtSemiLRJetCombCalc;

  public:

    TtSemiEvtSolution();
    virtual ~TtSemiEvtSolution();

    // methods to get original TopObjects 
    TopJet                    getHadb() const;
    TopJet                    getHadp() const;
    TopJet                    getHadq() const;
    TopJet                    getLepb() const;
    TopMuon                   getMuon() const;
    TopElectron               getElectron() const;
    TopMET                    getNeutrino() const;
    // methods to get the MC matched particles
    const edm::RefProd<TtGenEvent> & getGenEvent() const;
    const reco::GenParticle *   getGenHadt() const;
    const reco::GenParticle *   getGenHadW() const;
    const reco::GenParticle *   getGenHadb() const;
    const reco::GenParticle *   getGenHadp() const;
    const reco::GenParticle *   getGenHadq() const;
    const reco::GenParticle *   getGenLept() const;
    const reco::GenParticle *   getGenLepW() const;
    const reco::GenParticle *   getGenLepb() const;
    const reco::GenParticle *   getGenLepl() const;
    const reco::GenParticle *   getGenLepn() const;
    // methods to get reconstructed objects 
    reco::Particle            getRecHadt() const;
    reco::Particle            getRecHadW() const;       
    TopJetType                getRecHadb() const;
    TopJetType                getRecHadp() const;
    TopJetType                getRecHadq() const;
    reco::Particle            getRecLept() const;             
    reco::Particle            getRecLepW() const;  
    TopJetType                getRecLepb() const; 
    TopMuon                   getRecLepm() const;
    TopElectron               getRecLepe() const;
    TopMET                    getRecLepn() const;  
    // FIXME: Why these functions??? Not needed!
    // methods to get calibrated objects 
    reco::Particle            getCalHadt() const;
    reco::Particle            getCalHadW() const;
    TopJet                    getCalHadb() const;
    TopJet                    getCalHadp() const;
    TopJet                    getCalHadq() const;
    reco::Particle            getCalLept() const;
    reco::Particle            getCalLepW() const;
    TopJet                    getCalLepb() const;
    TopMuon                   getCalLepm() const;
    TopElectron               getCalLepe() const;
    TopMET                    getCalLepn() const;
    // methods to get fitted objects
    reco::Particle            getFitHadt() const;
    reco::Particle            getFitHadW() const;
    TopParticle               getFitHadb() const;
    TopParticle               getFitHadp() const;
    TopParticle               getFitHadq() const;
    reco::Particle            getFitLept() const;      
    reco::Particle            getFitLepW() const;
    TopParticle               getFitLepb() const;
    TopParticle               getFitLepl() const; 
    TopParticle               getFitLepn() const;    
    // method to get the selected semileptonic decay chain 
    std::string               getDecay() const                 { return decay_; }
    // methods to get info on the matching
    double                    getMCBestSumAngles() const       { return sumAnglejp_; };
    double                    getMCBestAngleHadp() const       { return angleHadp_; };
    double                    getMCBestAngleHadq() const       { return angleHadq_; };
    double                    getMCBestAngleHadb() const       { return angleHadb_; };
    double                    getMCBestAngleLepb() const       { return angleLepb_; };
    int                       getMCChangeWQ() const            { return changeWQ_; };     
    // methods to get the selected kinfit parametrisations of each type of object 
    int                       getJetParametrisation() const      { return jetParam_; }
    int                       getLeptonParametrisation() const   { return leptonParam_; }
    int                       getNeutrinoParametrisation() const { return neutrinoParam_; }
    // method to get the prob. of the chi2 value resulting from the kinematic fit
    double                    getProbChi2() const              { return probChi2_; }
    // methods to get info on the outcome of the signal selection LR
    double                    getLRSignalEvtObsVal(unsigned int) const;
    double                    getLRSignalEvtLRval() const      { return lrSignalEvtLRval_; }
    double                    getLRSignalEvtProb() const       { return lrSignalEvtProb_; }
    // methods to get info on the outcome of the different jet combination methods
    int                       getMCBestJetComb() const         { return mcBestJetComb_; }
    int                       getSimpleBestJetComb() const     { return simpleBestJetComb_; }
    int                       getLRBestJetComb() const         { return lrBestJetComb_; }
    double                    getLRJetCombObsVal(unsigned int) const;
    double                    getLRJetCombLRval() const        { return lrJetCombLRval_; }
    double                    getLRJetCombProb() const         { return lrJetCombProb_; }

  protected:         

    // method to set the generated event
    void                      setGenEvt(const edm::Handle<TtGenEvent> & aGenEvt);
    // methods to set the basic TopObjects
    void                      setJetCorrectionScheme(int jetCorrScheme);
    void                      setHadp(const edm::Handle<std::vector<TopJet> > & jh, int i);
    void                      setHadq(const edm::Handle<std::vector<TopJet> > & jh, int i);
    void                      setHadb(const edm::Handle<std::vector<TopJet> > & jh, int i);
    void                      setLepb(const edm::Handle<std::vector<TopJet> > & jh, int i);
    void                      setMuon(const edm::Handle<std::vector<TopMuon> > & mh, int i);
    void                      setElectron(const edm::Handle<std::vector<TopElectron> > & eh, int i);
    void                      setNeutrino(const edm::Handle<std::vector<TopMET> > & nh, int i);
    // methods to set the fitted objects 
    void                      setFitHadb(const TopParticle & aFitHadb);
    void                      setFitHadp(const TopParticle & aFitHadp);
    void                      setFitHadq(const TopParticle & aFitHadq);
    void                      setFitLepb(const TopParticle & aFitLepb);
    void                      setFitLepl(const TopParticle & aFitLepl);
    void                      setFitLepn(const TopParticle & aFitLepn);
    // methods to set the info on the matching
    void                      setMCBestSumAngles(double sdr);
    void                      setMCBestAngleHadp(double adr);
    void                      setMCBestAngleHadq(double adr);
    void                      setMCBestAngleHadb(double adr);
    void                      setMCBestAngleLepb(double adr);
    void                      setMCChangeWQ(int wq);
    // methods to set the kinfit parametrisations of each type of object 
    void                      setJetParametrisation(int jp);
    void                      setLeptonParametrisation(int lp);
    void                      setNeutrinoParametrisation(int mp);
    // method to set the prob. of the chi2 value resulting from the kinematic fit 
    void                      setProbChi2(double c);
    // methods to set the outcome of the different jet combination methods
    void                      setMCBestJetComb(int mcbs);
    void                      setSimpleBestJetComb(int sbs);
    void                      setLRBestJetComb(int lrbs);
    void                      setLRJetCombObservables(std::vector<std::pair<unsigned int, double> > varval);
    void                      setLRJetCombLRval(double clr);
    void                      setLRJetCombProb(double plr);
    // methods to set the outcome of the signal selection LR
    void                      setLRSignalEvtObservables(std::vector<std::pair<unsigned int, double> > varval);
    void                      setLRSignalEvtLRval(double clr);
    void                      setLRSignalEvtProb(double plr);

  private:

    // particle content
    edm::RefProd<TtGenEvent>            theGenEvt_;
    edm::Ref<std::vector<TopJet> >      hadb_, hadp_, hadq_, lepb_;
    edm::Ref<std::vector<TopMuon> >     muon_;
    edm::Ref<std::vector<TopElectron> > electron_;
    edm::Ref<std::vector<TopMET> >      neutrino_;
    std::vector<TopParticle>            fitHadb_, fitHadp_, fitHadq_;
    std::vector<TopParticle>            fitLepb_, fitLepl_, fitLepn_;
    // miscellaneous
    std::string               decay_;
    int                       jetCorrScheme_;
    double                    sumAnglejp_, angleHadp_, angleHadq_, angleHadb_, angleLepb_;
    int                       changeWQ_;
    int                       jetParam_, leptonParam_, neutrinoParam_;
    double                    probChi2_;
    int                       mcBestJetComb_, simpleBestJetComb_, lrBestJetComb_;
    double                    lrJetCombLRval_, lrJetCombProb_;
    double                    lrSignalEvtLRval_, lrSignalEvtProb_;
    std::vector<std::pair<unsigned int, double> > lrJetCombVarVal_;
    std::vector<std::pair<unsigned int, double> > lrSignalEvtVarVal_;

};


#endif
