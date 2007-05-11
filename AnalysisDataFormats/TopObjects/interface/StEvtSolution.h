#ifndef TopObjects_StEvtSolution_h
#define TopObjects_StEvtSolution_h
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "StGenEvent.h"
#include "TopJetObject.h"
#include "TopMuonObject.h"
#include "TopElectronObject.h"
#include "TopMETObject.h"
#include <vector>

using namespace reco;

class StEvtSolution
{
   public:
      StEvtSolution();
      virtual ~StEvtSolution();
      
      //      void setHadp(TopJetObject);
      //      void setHadq(TopJetObject);
      //      void setHadb(TopJetObject);
      //      void setLepb(TopJetObject);
      void setBottom(TopJetObject);
      void setLight(TopJetObject);
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

      
      TopJetObject         	getBottom() const  		{ return bottom; };
      TopJetObject         	getLight() const  		{ return light; };
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
      
      Particle 			getGenBottom() const		{ return genBottom; };
      Particle 			getGenLight() const		{ return genLight; };
      Particle 			getGenLepl() const		{ return genLepl; };
      Particle 			getGenLepn() const		{ return genLepn; };
      Particle 			getGenLepW() const		{ return genLepW; };
      Particle 			getGenLept() const		{ return genLept; };
      double 			getSumDeltaRjp() const		{ return sumDeltaRjp; };
      double 			getDeltaRhadp() const		{ return deltaRhadp; };
      double 			getDeltaRhadq() const		{ return deltaRhadq; };
      double 			getDeltaRhadb() const		{ return deltaRhadb; };
      double 			getDeltaRlepb() const		{ return deltaRlepb; };
      int			getChangeWQ() const		{ return changeWQ; };
      bool			getBestSol() const		{ return bestSol; };
      
      JetType		        getRecBottom() const;
      JetType                   getRecLight() const;
      TopMuon                    getRecLepm() const;
      TopElectron                getRecLepe() const;
      TopMET                     getRecLepn() const;  
      Particle                  getRecLepW() const;  
      Particle                  getRecLept() const;
      
      TopJet                     getCalBottom() const;
      TopJet                     getCalLight() const;
      Particle                  getCalLept() const;
      
      TopParticle                getFitBottom() const;
      TopParticle                getFitLight() const;
      TopParticle                getFitLepm() const; 
      TopParticle                getFitLepe() const;      
      TopParticle                getFitLepn() const;    
      Particle	                getFitLepW() const;
      Particle	                getFitLept() const;
        
   private:
      Particle         		genBottom, genLight, genLepl, genLepn, genLepW, genLept;
      TopJetObject         	bottom, light;
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
