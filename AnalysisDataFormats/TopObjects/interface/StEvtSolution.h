//
// $Id: StEvtSolution.h,v 1.9 2007/11/24 11:03:15 lowette Exp $
//

#ifndef TopObjects_StEvtSolution_h
#define TopObjects_StEvtSolution_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"

#include "AnalysisDataFormats/TopObjects/interface/StGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMuon.h"
#include "AnalysisDataFormats/TopObjects/interface/TopElectron.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMET.h"

#include <vector>


class StEvtSolution {

  friend class StEvtSolutionMaker;
  friend class StKinFitter;

  public:

    StEvtSolution();
    virtual ~StEvtSolution();
    
    // methods to get original TopObjects 
    TopJet         getBottom()   const;
    TopJet         getLight()    const;
    TopMuon        getMuon()     const;
    TopElectron    getElectron() const;
    TopMET         getNeutrino() const;
    reco::Particle getLepW()     const;  
    reco::Particle getLept()     const;
    // methods to get the MC matched particles
    const edm::RefProd<StGenEvent> & getGenEvent() const;
    const reco::GenParticle * getGenBottom()   const;
//    const reco::GenParticle * getGenLight()    const; // not implemented yet
    const reco::GenParticle * getGenLepton()   const;
    const reco::GenParticle * getGenNeutrino() const;
    const reco::GenParticle * getGenLepW()     const;
    const reco::GenParticle * getGenLept()     const;
    // methods to get reconstructed objects
    TopJetType     getRecBottom()   const;
    TopJetType     getRecLight()    const;
    TopMuon        getRecMuon()     const; // redundant
    TopElectron    getRecElectron() const; // redundant
    TopMET         getRecNeutrino() const; // redundant
    reco::Particle getRecLepW()     const; // redundant
    reco::Particle getRecLept()     const;
    // methods to get fitted objects
    TopParticle    getFitBottom()   const;
    TopParticle    getFitLight()    const;
    TopParticle    getFitLepton()   const;
    TopParticle    getFitNeutrino() const;
    reco::Particle getFitLepW()     const;
    reco::Particle getFitLept()     const;
    // method to get the info on the selected decay
    std::string         getDecay()          const { return decay_; }
    // methods to get other info on the event
    double              getChi2Prob()       const { return chi2Prob_; }
    std::vector<double> getScanValues()     const { return scanValues_; }
    double              getPtrueCombExist() const { return pTrueCombExist_; }
    double              getPtrueBJetSel()   const { return pTrueBJetSel_; }
    double              getPtrueBhadrSel()  const { return pTrueBhadrSel_; }
    double              getPtrueJetComb()   const { return pTrueJetComb_; }
    double              getSignalPur()      const { return signalPur_; }
    double              getSignalLRTot()    const { return signalLRTot_; }
    double              getSumDeltaRjp()    const { return sumDeltaRjp_; }
    double              getDeltaRB()        const { return deltaRB_; }
    double              getDeltaRL()        const { return deltaRL_; }
    int                 getChangeBL()       const { return changeBL_; }
    bool                getBestSol()        const { return bestSol_; }

  protected:         

    // method to set the generated event
    void setGenEvt(const edm::Handle<StGenEvent> & aGenEvt);
    // methods to set the basic TopObjects
    void setJetCorrectionScheme(int jetCorrScheme);
    void setBottom(const edm::Handle<std::vector<TopJet> > & jh, int i);
    void setLight(const edm::Handle<std::vector<TopJet> > & jh, int i);
    void setMuon(const edm::Handle<std::vector<TopMuon> > & mh, int i);
    void setElectron(const edm::Handle<std::vector<TopElectron> > & eh, int i);
    void setNeutrino(const edm::Handle<std::vector<TopMET> > & nh, int i);
    // methods to set the fitted objects 
    void setFitBottom(const TopParticle & aFitBottom);
    void setFitLight(const TopParticle & aFitLight);
    void setFitLepton(const TopParticle & aFitLepton);
    void setFitNeutrino(const TopParticle & aFitNeutrino);
    // methods to set other info on the event
    void setChi2Prob(double c);
    void setScanValues(const std::vector<double> & v);
    void setPtrueCombExist(double pce);
    void setPtrueBJetSel(double pbs);
    void setPtrueBhadrSel(double pbh);
    void setPtrueJetComb(double pt);
    void setSignalPurity(double c);
    void setSignalLRTot(double c);
    void setSumDeltaRjp(double);
    void setDeltaRB(double);
    void setDeltaRL(double);
    void setChangeBL(int);
    void setBestSol(bool);

  private:

    // particle content
    edm::RefProd<StGenEvent>            theGenEvt_;
    edm::Ref<std::vector<TopJet> >      bottom_, light_;
    edm::Ref<std::vector<TopMuon> >     muon_;
    edm::Ref<std::vector<TopElectron> > electron_;
    edm::Ref<std::vector<TopMET> >      neutrino_;
    std::vector<TopParticle>            fitBottom_, fitLight_, fitLepton_, fitNeutrino_;
    // miscellaneous
    std::string         decay_;
    int                 jetCorrScheme_;
    double              chi2Prob_;
    std::vector<double> scanValues_;
    double              pTrueCombExist_, pTrueBJetSel_, pTrueBhadrSel_, pTrueJetComb_;
    double              signalPur_, signalLRTot_;
    double              sumDeltaRjp_, deltaRB_, deltaRL_;
    int                 changeBL_;
    bool                bestSol_;
//    double              jetMatchPur_;

};


#endif
