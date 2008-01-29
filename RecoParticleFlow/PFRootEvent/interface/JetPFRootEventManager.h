#ifndef Demo_PFRootEvent_JetPFRootEventManager_h
#define Demo_PFRootEvent_JetPFRootEventManager_h

#include "RecoParticleFlow/PFRootEvent/interface/PFRootEventManager.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/BasicJetfwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetfwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetfwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h" 
#include "RecoJets/JetAlgorithms/interface/CMSIterativeConeAlgorithm.h"
#include "RecoJets/JetAlgorithms/interface/JetRecoTypes.h"
class TTree;
class TBranch;
class TFile;
class TCanvas;
class TH2F;
class TGraph;
class IO;
class TH1F;
class Utils;
class FWLiteJetProducer;
/*!
\author Joanna Weng
\date July 2006
*/

class JetPFRootEventManager : public PFRootEventManager {
	
	public:
	/// default constructor
	JetPFRootEventManager(const char* file);
	/// destructor
	~JetPFRootEventManager();
	
	/// process one entry 
	bool processEntry(int entry);
	/// ????
	void write();  
	/// reset vectors before next event
	void reset(); 
	/// print jet relevant parameters	
	void print();
	/// read options from .opt file
	void readOptions(const char* file, bool refresh=true);
	/// read data from simulation tree
	bool readFromSimulation(int entry);
	//Read the CMSSW jets from simulation tree 
	void readCMSSWJets();
	void makeGenJets();   
	void makeCaloJets();   
	void makePFJets();
	///Make jets from PFCandidtes and Calotowers in simulation tree 
	void makeFWLiteJets(const reco::CandidateCollection& Candidates);
	
	
	/// Reconstructed CMSSW jets from Particle Flow
	TBranch*   recPFJetsBranch_;
	/// Reconstructed CMSSW jets from Calo towers
	TBranch*   recCaloJetsBranch_;	
	/// Gen Particles Candidates
	TBranch* genParticleCandBranch_;
	/// Calotower Candidates
	TBranch* recCaloTowersCandBranch_;
	/// ParticleFlow Candidates
	TBranch* recParticleFlowCandBranch_;  
		/// Vector to read in CMSSW Gen candidates
	std::auto_ptr<reco::CandidateCollection> genParticleCand_;
	/// Vector to read in CMSSW calo tower candidates
	std::auto_ptr<reco::CandidateCollection> caloTowersCand_; 
	/// Vector to read in CMSSW PF candidates
	std::auto_ptr<reco::CandidateCollection> particleFlowCand_;
	/// Vector to read in CMSSW calo jets
	std::auto_ptr<std::vector<reco::CaloJet> >  reccalojets_;
	/// Vector to read in CMSSW PFJets
	std::auto_ptr<std::vector<reco::PFJet> >recpfjets_;  
	std::auto_ptr<reco::CandidateCollection> test_;
	/// options file parser 
	IO*        options_;
	
	// jets parameters ----------------------------------------
	/// Minimum ET for jet constituents
	double mEtInputCut_;
	/// Minimum energy for jet constituents
	double mEInputCut_ ;
	/// Seed to start jet reconstruction
	double seedThreshold_;
	/// Radius of the cone
	double coneRadius_;
	/// Fraction of (alowed) overlapping
	double coneAreaFraction_;
	/// ????
	int maxPairSize_;
	/// ????
	int maxIterations_;
	/// ????
	double overlapThreshold_;
	double ptMin_;
	double rparam_;
	int algoType_;
	bool   jetsDebugCMSSW_;
	FWLiteJetProducer* jetMaker_;
};

#endif
