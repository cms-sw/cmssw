#ifndef RecoParticleFlow_Benchmark_PFMETBenchmark_h
#define RecoParticleFlow_Benchmark_PFMETBenchmark_h

#include "RecoParticleFlow/Benchmark/interface/PFBenchmarkAlgo.h"

#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//#include "FWCore/ServiceRegistry/interface/Service.h" 
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1F.h"
#include "TH2F.h"
#include <string>
#include <TFile.h>


class PFMETBenchmark;

class TH1F;
class TH2F;

//class DQMStore; // CMSSW_2_X_X not needed here?

class PFMETBenchmark {
	
 public:
	
  PFMETBenchmark();
  virtual ~PFMETBenchmark();
	
  void setup(
	     std::string Filename,
	     bool debug, 
	     bool plotAgainstReco=0, 
             std::string benchmarkLabel_ = "ParticleFlow", 
	     DQMStore * dbe_store = NULL
	     );
  void process(const reco::PFMETCollection& , 
	       const reco::GenParticleCollection&, 
	       const reco::CaloMETCollection&, 
	       const reco::METCollection& );
  void calculateQuantities(const reco::PFMETCollection&, 
			   const reco::GenParticleCollection&, 
			   const reco::CaloMETCollection&,
			   const reco::METCollection&);
  void calculateQuantities(const reco::PFMETCollection&, 
			   const reco::GenParticleCollection&, 
			   const reco::CaloMETCollection&,
			   const reco::METCollection&,
			   const std::vector<reco::CaloJet>&,
			   const std::vector<reco::CaloJet>&);
  float getTrueMET(){return true_met;}
  float getTruePhi(){return true_phi;}
  float getTrueSET(){return true_set;}
  float getPFMET(){return rec_met;}
  float getPFMEX(){return rec_mex-true_mex;}
  float getPFMEY(){return rec_mey-true_mey;}
  float getPFPhi(){return rec_phi;}
  float getPFSET(){return rec_set;}
  float getCaloMET(){return calo_met;}
  float getCaloMEX(){return calo_mex-true_mex;}
  float getCaloMEY(){return calo_mey-true_mey;}
  float getCaloPhi(){return calo_phi;}
  float getCaloSET(){return calo_set;}
  float getTCMET(){return tc_met;}
  float getTCMEX(){return tc_mex-true_mex;}
  float getTCMEY(){return tc_mey-true_mey;}
  float getTCPhi(){return tc_phi;}
  float getTCSET(){return tc_set;}
  float getDeltaPFMET(){return rec_met - true_met;}
  float getDeltaPFPhi(){return mpi_pi(rec_phi - true_phi);}
  float getDeltaPFSET(){return rec_set - true_set;}
  float getDeltaCaloMET(){return calo_met - true_met;}
  float getDeltaCaloPhi(){return mpi_pi(calo_phi - true_phi);}
  float getDeltaCaloSET(){return calo_set - true_set;}
  float getDeltaTCMET(){return tc_met - true_met;}
  float getDeltaTCPhi(){return mpi_pi(tc_phi - true_phi);}
  float getDeltaTCSET(){return tc_set - true_set;}
  double mpi_pi(double angle);
  void analyse();
  //void FitSlicesInY(TH2F*, TH1F*, TH1F*, bool, int);
  void write();
	
 private:
		
  TFile *file_;
	
  // histograms
  // delta Pt or E quantities for Barrel
  TProfile* profileSETvsSETresp;
  TProfile* profileMETvsMETresp;
  TH2F* hSETvsDeltaSET;
  TH2F* hSETvsDeltaMET;
  TH1F* hMEX;
  TH1F* hDeltaMEX;
  TH1F* hDeltaMET;
  TH1F* hDeltaPhi;
  TH1F* hDeltaSET;

  TH2F* hDeltaPFMETvstrueMET;
  TH2F* hDeltaCaloMETvstrueMET;
  TH2F* hDeltaPFPhivstrueMET;
  TH2F* hDeltaCaloPhivstrueMET;
  TH2F* hCaloMETvstrueMET;
  TH2F* hPFMETvstrueMET;
  TH2F* hDeltaCaloMEXvstrueSET;
  TH2F* hDeltaPFMEXvstrueSET;
  TH1F* hTrueMET;
  TH1F* hCaloMET;
  TH1F* hPFMET;
  TH1F* hTCMET;
  TH2F* hDeltaTCMETvstrueMET;
  TH2F* hDeltaTCPhivstrueMET;

  TProfile* profileCaloSETvsCaloSETresp;
  TProfile* profileCaloMETvsCaloMETresp;
  TH2F* hCaloSETvsDeltaCaloSET;
  TH2F* hCaloSETvsDeltaCaloMET;
  TH1F* hCaloMEX;
  TH1F* hDeltaCaloMEX;
  TH1F* hDeltaCaloMET;
  TH1F* hDeltaCaloPhi;
  TH1F* hDeltaCaloSET;

  TProfile* profileTCSETvsTCSETresp;
  TProfile* profileTCMETvsTCMETresp;
  TH2F* hTCSETvsDeltaTCSET;
  TH2F* hTCSETvsDeltaTCMET;
  TH1F* hTCMEX;
  TH1F* hDeltaTCMEX;
  TH1F* hDeltaTCMET;
  TH1F* hDeltaTCPhi;
  TH1F* hDeltaTCSET;

  TH2F* hDeltaTCMEXvstrueSET;
  TH2F* hTCMETvstrueMET;

  //TH1F *hmeanPF;   
  //TH1F *hmeanCalo; 
  //TH1F *hsigmaPF;  
  //TH1F *hsigmaCalo;
  //TH1F *hrmsPF;    
  //TH1F *hrmsCalo;  

  std::string outputFile_;	

  double true_set;
  double true_met;
  double true_mex;
  double true_mey;
  double true_phi;
  double rec_met;
  double rec_mex;
  double rec_mey;
  double rec_phi;
  double rec_set;
  double calo_met;
  double calo_mex;
  double calo_mey;
  double calo_phi;
  double calo_set;
  double tc_met;
  double tc_mex;
  double tc_mey;
  double tc_phi;
  double tc_set;

 protected:
		
  PFBenchmarkAlgo *algo_;
  bool debug_;
  bool plotAgainstReco_;
  DQMStore *dbe_;
};

#endif // RecoParticleFlow_Benchmark_PFMETBenchmark_h
