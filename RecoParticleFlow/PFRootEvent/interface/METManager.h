#ifndef __RecoParticleFlow_PFRootEvent_METManager__
#define __RecoParticleFlow_PFRootEvent_METManager__

#include "RecoParticleFlow/Benchmark/interface/GenericBenchmark.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

class METManager {

 public:

  METManager(std::string outmetfilename);
  
  /// setup a genericBenchmark
  void addGenBenchmark(std::string GenBenchmarkName);

  /// Fill the genericBenchmark histograms
  /// string = name of the folder in the root file (=name of the GenBenchmark)
  void FillHisto(std::string);

  /// Write the output root file of the genericBenchmark
  void write();

  void setMET1(const reco::MET*);
  void setMET2(const reco::MET*);

  /// set METX with the GenMet computed from the GenParticleCollection with computeGenMET(...)
  void setMET1(const reco::GenParticleCollection*);
  void setMET2(const reco::GenParticleCollection*);

  /// set METX with the pfMet recomputed from the pfCandidates_ with recomputePFMET(...)
  void setMET1(const reco::PFCandidateCollection&);
  void setMET2(const reco::PFCandidateCollection&);

  /// cout events in tail of Delta(MET1,MET2)
  void coutTailEvents(const int entry, const double DeltaMETcut,
		      const double DeltaPhicut, const double MET1cut) const;

  /// propagate the Jet Energy Corrections to the MET
  void propagateJECtoMET1(const std::vector<reco::CaloJet> caloJets,
		    const std::vector<reco::CaloJet> corr_caloJets);
  void propagateJECtoMET2(const std::vector<reco::CaloJet> caloJets,
		    const std::vector<reco::CaloJet> corr_caloJets);

  void SetIgnoreParticlesIDs(const std::vector<unsigned int>*);
  void SetSpecificIdCut(const std::vector<unsigned int>*, const std::vector<double>*);

 private:
 
  /// private functions
  reco::MET computeGenMET( const reco::GenParticleCollection* ) const;
  reco::MET recomputePFMET( const reco::PFCandidateCollection&) const;
  reco::MET propagateJEC(const reco::MET&, const std::vector<reco::CaloJet> caloJets,
			 const std::vector<reco::CaloJet> corr_caloJets) const;

  /// data members
  reco::MET MET1_;
  reco::MET MET2_;
  /// map of GenericBenchmarks, the key is his name
  std::map<std::string,GenericBenchmark> GenBenchmarkMap_;
  std::string outmetfilename_;
  TFile* outfile_;
  std::vector<unsigned int> vIgnoreParticlesIDs_;
  std::vector<unsigned int> trueMetSpecificIdCut_;
  std::vector<double> trueMetSpecificEtaCut_;

};

#endif
