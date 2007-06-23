#ifndef TopObjects_TtSemiEvtSolution_h
#define TopObjects_TtSemiEvtSolution_h
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "TtGenEvent.h"
#include "TopJet.h"
#include "TopLepton.h"
#include "TopMET.h"

class TtSemiEvtSolution
{
   friend class TtSemiEvtSolutionMaker;
   friend class TtSemiLRSignalSelObservables;
   friend class TtSemiLRSignalSelCalc;
   friend class TtSemiLRJetCombObservables;
   friend class TtSemiLRJetCombCalc;
   friend class TtSemiKinFitterEMom;
   friend class TtSemiKinFitterEtEtaPhi;
   friend class TtSemiKinFitterEtThetaPhi;
   
   public:
      TtSemiEvtSolution();
      virtual ~TtSemiEvtSolution();     
     
      // members to get original TopObjects 
      TopJet         		getHadp() const  		{ return hadp; };
      TopJet         		getHadq() const  		{ return hadq; };
      TopJet         		getHadb() const  		{ return hadb; };
      TopJet         		getLepb() const  		{ return lepb; };
      TopMuon	  		getMuon() const  		{ return muon; };
      TopElectron  		getElectron() const  		{ return electron; };
      TopMET    	  	getMET() const  		{ return met; }; 
 
      // members to get reconstructed objects 
      TopJetType                getRecHadp() const;
      TopJetType                getRecHadq() const;
      TopJetType                getRecHadb() const;
      TopJetType                getRecLepb() const; 
      TopMuon                   getRecLepm() const;
      TopElectron               getRecLepe() const;
      TopMET                    getRecLepn() const;  
      reco::Particle            getRecLepW() const;  
      reco::Particle            getRecHadW() const;       
      reco::Particle            getRecHadt() const;
      reco::Particle            getRecLept() const;      	
	
      // members to get calibrated objects 
      TopJet                    getCalHadp() const;
      TopJet                    getCalHadq() const;
      TopJet                    getCalHadb() const;
      TopJet                    getCalLepb() const;
      reco::Particle            getCalHadW() const; 
      reco::Particle            getCalHadt() const;
      reco::Particle            getCalLept() const;      	
	
      // members to get fitted objects 
      TopParticle               getFitHadp() const;
      TopParticle               getFitHadq() const;
      TopParticle               getFitHadb() const;
      TopParticle               getFitLepb() const;
      TopParticle               getFitLepm() const; 
      TopParticle               getFitLepe() const;      
      TopParticle               getFitLepn() const;    
      reco::Particle 	        getFitHadW() const;
      reco::Particle	        getFitLepW() const;
      reco::Particle	        getFitHadt() const;
      reco::Particle	        getFitLept() const;      
      
      // members to get the MC matched particles and info on the matching itself
      reco::Particle 		getGenHadp() const		{ return genHadp; };
      reco::Particle 		getGenHadq() const		{ return genHadq; };
      reco::Particle 		getGenHadb() const		{ return genHadb; };
      reco::Particle 		getGenLepb() const		{ return genLepb; };
      reco::Particle 		getGenLepl() const		{ return genLepl; };
      reco::Particle 		getGenLepn() const		{ return genLepn; };
      reco::Particle 		getGenHadW() const		{ return genHadW; };
      reco::Particle 		getGenLepW() const		{ return genLepW; };
      reco::Particle 		getGenHadt() const		{ return genHadt; };
      reco::Particle 		getGenLept() const		{ return genLept; };
      double 			getSumDeltaRjp() const		{ return sumDeltaRjp; };
      double 			getDeltaRhadp() const		{ return deltaRhadp; };
      double 			getDeltaRhadq() const		{ return deltaRhadq; };
      double 			getDeltaRhadb() const		{ return deltaRhadb; };
      double 			getDeltaRlepb() const		{ return deltaRlepb; };
      int			getChangeWQ() const		{ return changeWQ; };      
      // member to get the selected semileptonic decay chain 
      std::string  		getDecay() const  		{ return decay; };      
      
      // members to get the selected kinfit parametrisations of each type of object 
      int    			getJetParametrisation() const   { return jetparam; };
      int    			getLeptonParametrisation() const{ return leptonparam; };
      int    			getMETParametrisation() const   { return metparam; };    
      
      // member to get the prob. of the chi2 value resulting from the kinematic fit
      double 	    	  	getProbChi2() const 		{ return probChi2; };      
      
      // members to get info on the outcome of the signal selection LR
      double                    getLRSignalEvtObsVal(unsigned int) const;
      double                    getLRSignalEvtLRval() const 	{return lrSignalEvtLRval;};
      double                    getLRSignalEvtProb() const 	{return lrSignalEvtProb;};
      
      // members to get info on the outcome of the different jet combination methods
      int			getMCCorrJetComb() const	{ return mcCorrJetComb; };
      int			getSimpleCorrJetComb() const	{ return simpleCorrJetComb; };
      int			getLRCorrJetComb() const	{ return lrCorrJetComb; };      
      double                    getLRJetCombObsVal(unsigned int) const;
      double                    getLRJetCombLRval() const 	{return lrJetCombLRval;};
      double                    getLRJetCombProb() const 	{return lrJetCombProb;};
      
   protected:         
      // members to set the TopObjects
      void 			setHadp(TopJet);
      void 			setHadq(TopJet);
      void 			setHadb(TopJet);
      void 			setLepb(TopJet);
      void 			setMuon(TopMuon);
      void 			setElectron(TopElectron);
      void 			setMET(TopMET);
      
      // members to set the MC matched particles and info on the matching itself
      void 			setGenEvt(const TtGenEvent&);
      void 			setSumDeltaRjp(double);
      void			setDeltaRhadp(double);
      void 			setDeltaRhadq(double);
      void 			setDeltaRhadb(double);
      void 			setDeltaRlepb(double);
      void 			setChangeWQ(int);
      
      // members to set the kinfit parametrisations of each type of object 
      void 			setJetParametrisation(int);
      void 			setLeptonParametrisation(int);
      void 			setMETParametrisation(int);
      
      // members to set the prob. of the chi2 value resulting from the kinematic fit 
      void 			setProbChi2(double);
            
      // members to set the outcome of the signal selection LR
      void 			setLRSignalEvtObservables(std::vector<std::pair<unsigned int, double> >);
      void 			setLRSignalEvtLRval(double);
      void 			setLRSignalEvtProb(double);
      
      // members to set the outcome of the different jet combination methods
      void 			setMCCorrJetComb(int);
      void 			setSimpleCorrJetComb(int);
      void 			setLRCorrJetComb(int);
      void 			setLRJetCombObservables(std::vector<std::pair<unsigned int, double> >);
      void 			setLRJetCombLRval(double);
      void 			setLRJetCombProb(double);
	
   private:
      reco::Particle	genHadp, genHadq, genHadb, genLepb, genLepl, genLepn, genHadW, genLepW, genHadt, genLept;
      TopJet         		hadp, hadq, hadb, lepb;
      TopMuon        		muon;
      TopElectron    		electron;
      TopMET 	    		met;
      std::string        	decay;
      double 	    		probChi2, lrJetCombLRval, lrJetCombProb, lrSignalEvtLRval, lrSignalEvtProb;
      double			sumDeltaRjp,deltaRhadp,deltaRhadq,deltaRhadb,deltaRlepb;
      int			mcCorrJetComb, simpleCorrJetComb, lrCorrJetComb;
      int 			changeWQ, jetparam, leptonparam, metparam;      
      std::vector<std::pair<unsigned int, double> > lrJetCombVarVal;
      std::vector<std::pair<unsigned int, double> > lrSignalEvtVarVal;
};

#endif
