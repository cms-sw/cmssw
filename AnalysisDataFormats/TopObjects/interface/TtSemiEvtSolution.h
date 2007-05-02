#ifndef TopObjects_TtSemiEvtSolution_h
#define TopObjects_TtSemiEvtSolution_h
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TtGenEvent.h"
#include "TopJetObject.h"
#include "TopMuonObject.h"
#include "TopElectronObject.h"
#include "TopMETObject.h"
#include <vector>

using namespace reco;

class TtSemiEvtSolution
{
   public:
      TtSemiEvtSolution();
      virtual ~TtSemiEvtSolution();
      
      void setHadp(TopJetObject);
      void setHadq(TopJetObject);
      void setHadb(TopJetObject);
      void setLepb(TopJetObject);
      void setMuon(TopMuonObject);
      void setElectron(TopElectronObject);
      void setMET(TopMETObject);
      void setChi2(double);
      void setScanValues(std::vector<double>);
      void setPtrueCombExist(double);
      void setPtrueBJetSel(double);
      void setPtrueBhadrSel(double);
      void setPtrueJetComb(double);
      void setSignalPurity(double);
      void setSignalLRtot(double);
      
      void setGenEvt(vector<Candidate *>);
      void setSumDeltaRjp(double);
      void setDeltaRhadp(double);
      void setDeltaRhadq(double);
      void setDeltaRhadb(double);
      void setDeltaRlepb(double);
      void setChangeWQ(int);
      void setBestSol(bool);

      
      TopJetObject         	getHadp() const  		{ return hadp; };
      TopJetObject         	getHadq() const  		{ return hadq; };
      TopJetObject         	getHadb() const  		{ return hadb; };
      TopJetObject         	getLepb() const  		{ return lepb; };
      TopMuonObject	  	getMuon() const  		{ return muon; };
      TopElectronObject  	getElectron() const  		{ return electron; };
      TopMETObject    	  	getMET() const  		{ return met; };
      double 	    	  	getChi2() const 		{ return chi2; };
      double 	    	  	getPtrueCombExist() const	{ return ptrueCombExist;};
      double 	    	  	getPtrueBJetSel() const		{ return ptrueBJetSel;};
      double 	    	  	getPtrueBhadrSel() const	{ return ptrueBhadrSel;};
      double 	    	  	getPtrueJetComb() const		{ return ptrueJetComb;};
      double 	    	  	getSignalPur() const		{ return signalPur; };
      double 	    	  	getSignalLRtot() const   	{ return signalLRtot; };
      std::vector<double> 	getScanValues() const 		{ return scanValues; };
      string     	  	getDecay() const		{ return decay; };      
      
      Particle 			getGenHadp() const		{ return genHadp; };
      Particle 			getGenHadq() const		{ return genHadq; };
      Particle 			getGenHadb() const		{ return genHadb; };
      Particle 			getGenLepb() const		{ return genLepb; };
      Particle 			getGenLepl() const		{ return genLepl; };
      Particle 			getGenLepn() const		{ return genLepn; };
      Particle 			getGenHadW() const		{ return genHadW; };
      Particle 			getGenLepW() const		{ return genLepW; };
      Particle 			getGenHadt() const		{ return genHadt; };
      Particle 			getGenLept() const		{ return genLept; };
      double 			getSumDeltaRjp() const		{ return sumDeltaRjp; };
      double 			getDeltaRhadp() const		{ return deltaRhadp; };
      double 			getDeltaRhadq() const		{ return deltaRhadq; };
      double 			getDeltaRhadb() const		{ return deltaRhadb; };
      double 			getDeltaRlepb() const		{ return deltaRlepb; };
      int			getChangeWQ() const		{ return changeWQ; };
      bool			getBestSol() const		{ return bestSol; };
      
      jetType		        getRecHadp() const;
      jetType                   getRecHadq() const;
      jetType                   getRecHadb() const;
      jetType                   getRecLepb() const; 
      TopMuon                    getRecLepm() const;
      TopElectron                getRecLepe() const;
      TopMET                     getRecLepn() const;  
      Particle                  getRecLepW() const;  
      Particle                  getRecHadW() const;       
      Particle                  getRecHadt() const;
      Particle                  getRecLept() const;
      
      TopJet                     getCalHadp() const;
      TopJet                     getCalHadq() const;
      TopJet                     getCalHadb() const;
      TopJet                     getCalLepb() const;
      Particle                  getCalHadW() const; 
      Particle                  getCalHadt() const;
      Particle                  getCalLept() const;
      
      TopParticle                getFitHadp() const;
      TopParticle                getFitHadq() const;
      TopParticle                getFitHadb() const;
      TopParticle                getFitLepb() const;
      TopParticle                getFitLepm() const; 
      TopParticle                getFitLepe() const;      
      TopParticle                getFitLepn() const;    
      Particle 	                getFitHadW() const;
      Particle	                getFitLepW() const;
      Particle	                getFitHadt() const;
      Particle	                getFitLept() const;
        
   private:
      Particle         		genHadp, genHadq, genHadb, genLepb, genLepl, genLepn, genHadW, genLepW, genHadt, genLept;
      TopJetObject         	hadp, hadq, hadb, lepb;
      TopMuonObject        	muon;
      TopElectronObject    	electron;
      TopMETObject 	    	met;
      string        		decay;
      double 	    		chi2, jetMatchPur, signalPur, ptrueCombExist, signalLRtot, ptrueBJetSel, ptrueBhadrSel, ptrueJetComb;
      double			sumDeltaRjp,deltaRhadp,deltaRhadq,deltaRhadb,deltaRlepb;
      bool			bestSol;
      int 			changeWQ;
      std::vector<double> 	scanValues;
};

#endif
