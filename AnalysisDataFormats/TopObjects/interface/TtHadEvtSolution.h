#ifndef TopObjects_TtHadEvtSolution_h
#define TopObjects_TtHadEvtSolution_h
//
// $Id: TtHadEvtSolution.h,v 1.01 2007/09/30 09:30:00 mfhansen Exp $
// adapted TtSemiEvtSolution.h,v 1.14 2007/07/06 03:07:47 lowette Exp 
// for fully hadronic channel

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/Candidate/interface/Particle.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"

#include <vector>
#include <string>


class TtHadEvtSolution {
  friend class TtHadEvtSolutionMaker;
  friend class TtHadKinFitter;
  /*
  friend class TtHadLRSignalSelObservables;
  friend class TtHadLRSignalSelCalc;
  friend class TtHadLRJetCombObservables;
  friend class TtHadLRJetCombCalc;
  */
  public:

    TtHadEvtSolution();
    virtual ~TtHadEvtSolution();     

    // methods to get original TopObjects 
    TopJet                    getHadb() const;
    TopJet                    getHadp() const;
    TopJet                    getHadq() const;
    TopJet                    getHadbbar() const;
    TopJet                    getHadj() const;
    TopJet                    getHadk() const;

    // methods to get the MC matched particles
    const TtGenEvent &        getGenEvent() const { return *theGenEvt_; }
    const reco::Candidate *   getGenHadb() const;
    const reco::Candidate *   getGenHadp() const;
    const reco::Candidate *   getGenHadq() const;
    const reco::Candidate *   getGenHadbbar() const;
    const reco::Candidate *   getGenHadj() const;
    const reco::Candidate *   getGenHadk() const;

    // methods to get reconstructed objects 
    reco::Particle            getRecHadt() const;
    reco::Particle            getRecHadtbar() const;
    reco::Particle            getRecHadW() const;     
    reco::Particle            getRecHadWbar() const;       

    TopJetType                getRecHadb() const;
    TopJetType                getRecHadp() const;
    TopJetType                getRecHadq() const;
    TopJetType                getRecHadbbar() const;
    TopJetType                getRecHadj() const;
    TopJetType                getRecHadk() const;

    // methods to get calibrated objects 
    reco::Particle            getCalHadt() const;
    reco::Particle            getCalHadtbar() const;
    reco::Particle            getCalHadW() const;
    reco::Particle            getCalHadWbar() const;
    TopJet                    getCalHadb() const;
    TopJet                    getCalHadp() const;
    TopJet                    getCalHadq() const;
    TopJet                    getCalHadbbar() const;
    TopJet                    getCalHadj() const;
    TopJet                    getCalHadk() const;

    // methods to get fitted objects
    reco::Particle            getFitHadt() const;
    reco::Particle            getFitHadtbar() const;
    reco::Particle            getFitHadW() const;
    reco::Particle            getFitHadWbar() const;
    TopParticle               getFitHadb() const;
    TopParticle               getFitHadp() const;
    TopParticle               getFitHadq() const;
    TopParticle               getFitHadbbar() const;
    TopParticle               getFitHadj() const;
    TopParticle               getFitHadk() const;

    // method to get the selected hadronic decay chain 
    std::string               getDecay() const                 { return decay_; }
    // methods to get info on the matching
    double                    getMCBestSumAngles() const       { return sumAnglejp_; };
    double                    getMCBestAngleHadp() const       { return angleHadp_; };
    double                    getMCBestAngleHadq() const       { return angleHadq_; };
    double                    getMCBestAngleHadb() const       { return angleHadb_; };
    double                    getMCBestAngleHadj() const       { return angleHadj_; };
    double                    getMCBestAngleHadk() const       { return angleHadk_; };
    double                    getMCBestAngleHadbbar() const    { return angleHadbbar_; };
    int                       getMCChangeW1Q() const           { return changeW1Q_; };     
    int                       getMCChangeW2Q() const           { return changeW2Q_;}; 
    // methods to get the selected kinfit parametrisations of each type of object 
    int                       getJetParametrisation() const    { return jetParam_; }
    // method to get the prob. of the chi2 value resulting from the kinematic fit
    // added chi2 for all fits
    double                    getProbChi2() const             { return probChi2_; }
    // methods to get info on the outcome of the signal selection LR
    double                    getLRSignalEvtObsVal(unsigned int) const;
    double                    getLRSignalEvtLRval() const      { return lrSignalEvtLRval_; }
    double                    getLRSignalEvtProb() const       { return lrSignalEvtProb_; }
    // methods to get info on the outcome of the different jet combination methods
    int                       getMCCorrJetComb() const         { return mcCorrJetComb_; }
    int                       getSimpleCorrJetComb() const     { return simpleCorrJetComb_; }
    int                       getLRCorrJetComb() const         { return lrCorrJetComb_; }
    double                    getLRJetCombObsVal(unsigned int) const;
    double                    getLRJetCombLRval() const        { return lrJetCombLRval_; }
    double                    getLRJetCombProb() const         { return lrJetCombProb_; }

    //  protected:      seem to cause compile error, check!!!

    // method to set the generated event
    void                      setGenEvt(const edm::Handle<TtGenEvent> & aGenEvt);
    // methods to set the basic TopObjects
    void                      setHadp(const edm::Handle<std::vector<TopJet> > & jh, int i);
    void                      setHadq(const edm::Handle<std::vector<TopJet> > & jh, int i);
    void                      setHadb(const edm::Handle<std::vector<TopJet> > & jh, int i);
    void                      setHadj(const edm::Handle<std::vector<TopJet> > & jh, int i);
    void                      setHadk(const edm::Handle<std::vector<TopJet> > & jh, int i);
    void                      setHadbbar(const edm::Handle<std::vector<TopJet> > & jh, int i);
    // methods to set the fitted objects 
    void                      setFitHadb(const TopParticle & aFitHadb);
    void                      setFitHadp(const TopParticle & aFitHadp);
    void                      setFitHadq(const TopParticle & aFitHadq);
    void                      setFitHadbbar(const TopParticle & aFitHadbbar);
    void                      setFitHadj(const TopParticle & aFitHadj);
    void                      setFitHadk(const TopParticle & aFitHadk);

    // methods to set the info on the matching
    void                      setMCBestSumAngles(double sdr);
    void                      setMCBestAngleHadp(double adr);
    void                      setMCBestAngleHadq(double adr);
    void                      setMCBestAngleHadb(double adr);
    void                      setMCBestAngleHadj(double adr);
    void                      setMCBestAngleHadk(double adr);
    void                      setMCBestAngleHadbbar(double adr);
    void                      setMCChangeW1Q(int w1q);
    void                      setMCChangeW2Q(int w2q);
    // methods to set the kinfit parametrisations of each type of object 
    void                      setJetParametrisation(int jp);
    // method to set the prob. of the chi2 value resulting from the kinematic fit 
    void                      setProbChi2(double c);
    // methods to set the outcome of the different jet combination methods
    void                      setMCCorrJetComb(int mcbs);
    void                      setSimpleCorrJetComb(int sbs);
    void                      setLRCorrJetComb(int lrbs);
    void                      setLRJetCombObservables(std::vector<std::pair<unsigned int, double> > varval);
    void                      setLRJetCombLRval(double clr);
    void                      setLRJetCombProb(double plr);
    // methods to set the outcome of the signal selection LR
    void                      setLRSignalEvtObservables(std::vector<std::pair<unsigned int, double> > varval);
    void                      setLRSignalEvtLRval(double clr);
    void                      setLRSignalEvtProb(double plr);

  private:

    edm::RefProd<TtGenEvent>            theGenEvt_;
    edm::Ref<std::vector<TopJet> >      hadb_, hadp_, hadq_, hadbbar_,hadj_, hadk_;
    std::vector<TopParticle>            fitHadb_, fitHadp_, fitHadq_, fitHadbbar_, fitHadj_, fitHadk_;

    std::string               decay_;
    double                    sumAnglejp_, angleHadp_, angleHadq_, angleHadb_, angleHadbbar_, angleHadj_ , angleHadk_;
    int                       changeW1Q_, changeW2Q_;
    int                       jetParam_;
    double                    probChi2_;
    int                       mcCorrJetComb_, simpleCorrJetComb_, lrCorrJetComb_;
    double                    lrJetCombLRval_, lrJetCombProb_;
    double                    lrSignalEvtLRval_, lrSignalEvtProb_;
    std::vector<std::pair<unsigned int, double> > lrJetCombVarVal_;
    std::vector<std::pair<unsigned int, double> > lrSignalEvtVarVal_;

};


#endif
