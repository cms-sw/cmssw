#ifndef RecoParticleFlow_Benchmark_PFTauElecRejectionBenchmark_h
#define RecoParticleFlow_Benchmark_PFTauElecRejectionBenchmark_h

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminatorByIsolation.h"
#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "TLorentzVector.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1F.h"
#include "TH2F.h"
#include <string>
#include <TFile.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

class PFTauElecRejectionBenchmark;

class TH1F;
class TH2F;

//class DQMStore; // CMSSW_2_X_X not needed here?

class PFTauElecRejectionBenchmark {
	
 public:
  
  PFTauElecRejectionBenchmark();
  virtual ~PFTauElecRejectionBenchmark();
  
  void setup(
	     std::string Filename,
	     std::string benchmarkLabel,
	     double maxDeltaR, 
	     double minRecoPt, 
	     double maxRecoAbsEta, 
	     double minMCPt, 
	     double maxMCAbsEta, 
	     std::string sGenMatchObjectLabel,
	     bool applyEcalCrackCut,
	     DQMStore * db_store);
  void process(edm::Handle<edm::HepMCProduct> mcevt, edm::Handle<reco::PFTauCollection> pfTaus, 
	       edm::Handle<reco::PFTauDiscriminator> pfTauIsoDiscr, 
	       edm::Handle<reco::PFTauDiscriminator> pfTauElecDiscr);
  void write();
	
 private:
		
  bool isInEcalCrack(double eta) const;

  TFile *file_;
  std::string outputFile_;	
  std::string benchmarkLabel_;
  double maxDeltaR_;
  double minMCPt_;
  double maxMCAbsEta_;
  double minRecoPt_;
  double maxRecoAbsEta_;
  std::string sGenMatchObjectLabel_;
  bool applyEcalCrackCut_;

  // histograms
  TH1F *hEoverP;
  TH1F *hHoverP;
  TH1F *hEmfrac;

  TH1F *hEoverP_barrel;
  TH1F *hHoverP_barrel;
  TH1F *hEmfrac_barrel;

  TH1F *hEoverP_endcap;
  TH1F *hHoverP_endcap;
  TH1F *hEmfrac_endcap;

  TH1F *hEoverP_preid0;
  TH1F *hHoverP_preid0;
  TH1F *hEmfrac_preid0;

  TH1F *hEoverP_preid1;
  TH1F *hHoverP_preid1;
  TH1F *hEmfrac_preid1;

  TH1F *hElecPreID;
  TH1F *hElecMVA;
  TH1F *hTauElecDiscriminant;

  TH2F *hHoPvsEoP;
  TH2F *hHoPvsEoP_preid0;
  TH2F *hHoPvsEoP_preid1;

  TH2F *hEmfracvsEoP;
  TH2F *hEmfracvsEoP_preid0;
  TH2F *hEmfracvsEoP_preid1;

  TH1F *hpfcand_deltaEta;
  TH1F *hpfcand_deltaEta_weightE;
  TH1F *hpfcand_deltaPhiOverQ;
  TH1F *hpfcand_deltaPhiOverQ_weightE;

  TH1F *hleadTk_pt;
  TH1F *hleadTk_eta;
  TH1F *hleadTk_phi;

  // to be filled yet!
  TH1F *hleadGsfTk_pt;
  TH1F *hleadGsfTk_eta;
  TH1F *hleadGsfTk_phi;

	
  std::vector<TLorentzVector> _GenObjects;

 protected:
		
  DQMStore *db_;
};

#endif // RecoParticleFlow_Benchmark_PFTauElecRejectionBenchmark_h
