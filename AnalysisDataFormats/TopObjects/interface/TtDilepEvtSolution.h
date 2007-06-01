#ifndef TopObjects_TtDilepEvtSolution_h
#define TopObjects_TtDilepEvtSolution_h
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TtGenEvent.h"
#include "TopJet.h"
#include "TopLepton.h"
#include "TopMET.h"
#include <vector>

using namespace reco;

class TtDilepEvtSolution
{
   public:
      TtDilepEvtSolution();
      virtual ~TtDilepEvtSolution();
      
      void setGenEvt(std::vector<Candidate *>);
      void setBestSol(bool);
      
      void setMuonLepp(TopMuon);
      void setMuonLepm(TopMuon);
      void setElectronLepp(TopElectron);
      void setElectronLepm(TopElectron);
      void setB(TopJet);
      void setBbar(TopJet);
      void setMET(TopMET);
      
      std::string getWpDecay() const		{ return WpDecay; }; 
      std::string getWmDecay() const		{ return WmDecay; };
      
      Particle getGenLepp() const {return genLepp;};
      Particle getGenN() const {return genN;};
      Particle getGenB() const {return genB;};
      Particle  getGenBbar() const {return genBbar;};
      Particle  getGenLepm() const {return genLepm;};
      Particle getGenNbar() const {return genNbar;};
      Particle  getGenWp() const {return genWp;};
      Particle  getGenWm() const {return genWm;};
      Particle  getGenT() const {return genT;};
      Particle  getGenTbar() const {return genTbar;};
      bool getBestSol() const		{ return bestSol; };
      
      JetType getRecJetB() const;
      JetType getRecJetBbar() const;
      TopMET getRecMET() const;
      
      Particle getRecLepp() const;
      Particle getRecLepm() const;

      TopElectron getElectronLepp() const {return elecLepp;};
      TopElectron getElectronLepm() const {return elecLepm;};
      TopMuon getMuonLepp() const {return muonLepp;};
      TopMuon getMuonLepm() const {return muonLepm;};
      TopJet getJetB() const {return jetB;};
      TopJet getJetBbar() const {return jetBbar;};
      TopMET getMET() const {return met;};
      
      TopJet getCalJetB() const;
      TopJet getCalJetBbar() const;


      double getRecTopMass() const {return topmass_;};
      double getRecWeightMax() const {return weightmax_;};
      void setRecTopMass(double);
      void setRecWeightMax(double);
      
      //TopMuon genLepp  
   private:
      Particle genLepp, genN, genB, genBbar, genLepm, genNbar, genWp, genWm, genT, genTbar;
      TopElectron elecLepp, elecLepm;
      TopMuon muonLepp, muonLepm;
      TopJet jetB, jetBbar;
      TopMET met;
      
      bool bestSol;
      
      double topmass_;
      double weightmax_;
      
      std::string WpDecay;
      std::string WmDecay;
      
};

#endif
