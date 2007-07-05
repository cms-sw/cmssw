#ifndef TopObjects_StEvtSolution_h
#define TopObjects_StEvtSolution_h
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "StGenEvent.h"
#include "TopJet.h"
#include "TopLepton.h"
#include "TopMET.h"
#include <vector>

class StEvtSolution
{
   public:
      StEvtSolution();
      virtual ~StEvtSolution();
      
      //      void setHadp(TopJet);
      //      void setHadq(TopJet);
      //      void setHadb(TopJet);
      //      void setLepb(TopJet);
      void setBottom(TopJet);
      void setLight(TopJet);
      void setMuon(TopMuon);
      void setElectron(TopElectron);
      void setMET(TopMET);
      void setChi2(double);
      void setScanValues(std::vector<double>);
      void setPtrueCombExist(double);
      void setPtrueBJetSel(double);
      void setPtrueBhadrSel(double);
      void setPtrueJetComb(double);
      void setSignalPurity(double);
      void setSignalLRtot(double);
      
      void setGenEvt(std::vector<reco::Candidate *>);
      void setSumDeltaRjp(double);
      void setDeltaRB(double);
      void setDeltaRL(double);
      void setChangeBL(int);
      void setBestSol(bool);

      
      TopJet         		getBottom() const  		{ return bottom; };
      TopJet         		getLight() const  		{ return light; };
      TopMuon	  		getMuon() const  		{ return muon; };
      TopElectron  		getElectron() const  		{ return electron; };
      TopMET    	  	getMET() const  		{ return met; };
      double 	    	  	getChi2() const 		{ return chi2; };
      double 	    	  	getPtrueCombExist() const	{ return ptrueCombExist;};
      double 	    	  	getPtrueBJetSel() const		{ return ptrueBJetSel;};
      double 	    	  	getPtrueBhadrSel() const	{ return ptrueBhadrSel;};
      double 	    	  	getPtrueJetComb() const		{ return ptrueJetComb;};
      double 	    	  	getSignalPur() const		{ return signalPur; };
      double 	    	  	getSignalLRtot() const   	{ return signalLRtot; };
      std::vector<double> 	getScanValues() const 		{ return scanValues; };
      std::string     	  	getDecay() const		{ return decay; };      
      
      reco::Particle 		getGenBottom() const		{ return genBottom; };
      reco::Particle 		getGenLight() const		{ return genLight; };
      reco::Particle 		getGenLepl() const		{ return genLepl; };
      reco::Particle 		getGenLepn() const		{ return genLepn; };
      reco::Particle 		getGenLepW() const		{ return genLepW; };
      reco::Particle 		getGenLept() const		{ return genLept; };
      double 			getSumDeltaRjp() const		{ return sumDeltaRjp; };
      double 			getDeltaRB() const		{ return deltaRB; };
      double 			getDeltaRL() const		{ return deltaRL; };
      int			getChangeBL() const		{ return changeBL; };
      bool			getBestSol() const		{ return bestSol; };
      
      TopJetType                getRecBottom() const;
      TopJetType                getRecLight() const;
      TopMuon                   getRecLepm() const;
      TopElectron               getRecLepe() const;
      TopMET                    getRecLepn() const;  
      reco::Particle            getRecLepW() const;  
      reco::Particle            getRecLept() const;
      
      TopJet                    getCalBottom() const;
      TopJet                    getCalLight() const;
      reco::Particle            getCalLept() const;

// FIXME FIXME FIXME
// fit members must become part of the final state object
/*
      TopParticle         	getFitBottom() const;
      TopParticle         	getFitLight() const;
      TopParticle         	getFitLepm() const; 
      TopParticle         	getFitLepe() const;      
      TopParticle         	getFitLepn() const;    
      reco::Particle	        getFitLepW() const;
      reco::Particle	        getFitLept() const;
*/

   private:
      reco::Particle         	genBottom, genLight, genLepl, genLepn, genLepW, genLept;
      TopJet         		bottom, light;
      TopMuon        		muon;
      TopElectron    		electron;
      TopMET 	    		met;
      std::string      		decay;
      double 	    		chi2, jetMatchPur, signalPur, ptrueCombExist, signalLRtot, ptrueBJetSel, ptrueBhadrSel, ptrueJetComb;
      double			sumDeltaRjp,deltaRB,deltaRL;
      bool			bestSol;
      int 			changeBL;
      std::vector<double> 	scanValues;
};

#endif
