#ifndef TopObjects_TtSemiEvtSolution_h
#define TopObjects_TtSemiEvtSolution_h
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TtGenEvent.h"
#include "TopJet.h"
#include "TopLepton.h"
#include "TopMET.h"
#include <vector>
#include "DataFormats/Candidate/interface/Particle.h"

class TtSemiEvtSolution
{
   public:
      TtSemiEvtSolution();
      virtual ~TtSemiEvtSolution();
      
      void setHadp(TopJet);
      void setHadq(TopJet);
      void setHadb(TopJet);
      void setLepb(TopJet);
      void setMuon(TopMuon);
      void setElectron(TopElectron);
      void setMET(TopMET);
      void setJetParametrisation(int);
      void setLeptonParametrisation(int);
      void setMETParametrisation(int);
      void setProbChi2(double);
      void setPtrueCombExist(double);
      void setPtrueBJetSel(double);
      void setPtrueBhadrSel(double);
      void setPtrueJetComb(double);
      void setSignalPurity(double);
      void setSignalLRtot(double);
      
      void setGenEvt(std::vector<reco::Candidate *>);
      void setSumDeltaRjp(double);
      void setDeltaRhadp(double);
      void setDeltaRhadq(double);
      void setDeltaRhadb(double);
      void setDeltaRlepb(double);
      void setChangeWQ(int);
      void setMCBestSol(int);
      void setSimpleBestSol(int);

      
      TopJet         			getHadp() const  		{ return hadp; };
      TopJet         			getHadq() const  		{ return hadq; };
      TopJet         			getHadb() const  		{ return hadb; };
      TopJet         			getLepb() const  		{ return lepb; };
      TopMuon	  			getMuon() const  		{ return muon; };
      TopElectron  			getElectron() const  		{ return electron; };
      std::string  			getDecay() const  		{ return decay; };
      TopMET    	  		getMET() const  		{ return met; };
      int    				getJetParametrisation() const   { return jetparam; };
      int    				getLeptonParametrisation() const{ return leptonparam; };
      int    				getMETParametrisation() const   { return metparam; };
      double 	    	  		getProbChi2() const 		{ return probChi2; };
      double 	    	  		getPtrueCombExist() const	{ return ptrueCombExist;};
      double 	    	  		getPtrueBJetSel() const		{ return ptrueBJetSel;};
      double 	    	  		getPtrueBhadrSel() const	{ return ptrueBhadrSel;};
      double 	    	  		getPtrueJetComb() const		{ return ptrueJetComb;};
      double 	    	  		getSignalPur() const		{ return signalPur; };
      double 	    	  		getSignalLRtot() const   	{ return signalLRtot; };
      
      reco::Particle 			getGenHadp() const		{ return genHadp; };
      reco::Particle 			getGenHadq() const		{ return genHadq; };
      reco::Particle 			getGenHadb() const		{ return genHadb; };
      reco::Particle 			getGenLepb() const		{ return genLepb; };
      reco::Particle 			getGenLepl() const		{ return genLepl; };
      reco::Particle 			getGenLepn() const		{ return genLepn; };
      reco::Particle 			getGenHadW() const		{ return genHadW; };
      reco::Particle 			getGenLepW() const		{ return genLepW; };
      reco::Particle 			getGenHadt() const		{ return genHadt; };
      reco::Particle 			getGenLept() const		{ return genLept; };
      double 				getSumDeltaRjp() const		{ return sumDeltaRjp; };
      double 				getDeltaRhadp() const		{ return deltaRhadp; };
      double 				getDeltaRhadq() const		{ return deltaRhadq; };
      double 				getDeltaRhadb() const		{ return deltaRhadb; };
      double 				getDeltaRlepb() const		{ return deltaRlepb; };
      int				getChangeWQ() const		{ return changeWQ; };
      int				getMCBestSol() const		{ return mcBestSol; };
      int				getSimpleBestSol() const	{ return simpleBestSol; };
      	
      JetType		        	getRecHadp() const;
      JetType                   	getRecHadq() const;
      JetType                   	getRecHadb() const;
      JetType                   	getRecLepb() const; 
      TopMuon                   	getRecLepm() const;
      TopElectron               	getRecLepe() const;
      TopMET                    	getRecLepn() const;  
      reco::Particle                  	getRecLepW() const;  
      reco::Particle                  	getRecHadW() const;       
      reco::Particle                  	getRecHadt() const;
      reco::Particle                  	getRecLept() const;
      	
      TopJet                    	getCalHadp() const;
      TopJet                    	getCalHadq() const;
      TopJet                    	getCalHadb() const;
      TopJet                    	getCalLepb() const;
      reco::Particle                  	getCalHadW() const; 
      reco::Particle                  	getCalHadt() const;
      reco::Particle                  	getCalLept() const;
      	
      TopParticle               	getFitHadp() const;
      TopParticle               	getFitHadq() const;
      TopParticle               	getFitHadb() const;
      TopParticle               	getFitLepb() const;
      TopParticle               	getFitLepm() const; 
      TopParticle               	getFitLepe() const;      
      TopParticle               	getFitLepn() const;    
      reco::Particle 	                getFitHadW() const;
      reco::Particle	                getFitLepW() const;
      reco::Particle	                getFitHadt() const;
      reco::Particle	                getFitLept() const;
        
   private:
      reco::Particle		        genHadp, genHadq, genHadb, genLepb, genLepl, genLepn, genHadW, genLepW, genHadt, genLept;
      TopJet         			hadp, hadq, hadb, lepb;
      TopMuon        			muon;
      TopElectron    			electron;
      TopMET 	    			met;
      std::string        		decay;
      double 	    			probChi2, jetMatchPur, signalPur, ptrueCombExist, signalLRtot, ptrueBJetSel, ptrueBhadrSel, ptrueJetComb;
      double				sumDeltaRjp,deltaRhadp,deltaRhadq,deltaRhadb,deltaRlepb;
      int				mcBestSol, simpleBestSol;
      int 				changeWQ, jetparam, leptonparam, metparam;
};

#endif
