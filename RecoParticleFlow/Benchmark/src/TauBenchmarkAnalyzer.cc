// system include files
#include <memory>

// user include files

#include "RecoParticleFlow/Benchmark/interface/TauBenchmarkAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"




#include "TH1.h"
#include "TFile.h"


#include "RecoParticleFlow/Benchmark/interface/PFBenchmarkAlgo.h"

using namespace std;

//define this as a plug-in
DEFINE_FWK_MODULE(TauBenchmarkAnalyzer);

TauBenchmarkAnalyzer::TauBenchmarkAnalyzer(const edm::ParameterSet& iConfig)
{

   outputRootFileName_ = iConfig.getUntrackedParameter< string >("outputRootFileName","tauBenchmark.root");
  caloJetsLabel_= iConfig.getUntrackedParameter< string >("recoCaloJetsLabel","iterativeCone5CaloJets");
  pfJetsLabel_= iConfig.getUntrackedParameter< string >("pfJetsLabel","iterativeCone5PFJets");
  genJetsLabel_= iConfig.getUntrackedParameter< string >("genCaloJetsLabel","iterativeCone5GenJets");

}

TauBenchmarkAnalyzer::~TauBenchmarkAnalyzer()
{
  
}


void TauBenchmarkAnalyzer::beginJob(const edm::EventSetup&)
{
  benchmark=new PFBenchmarkAlgo(outputRootFileName_);
  //benchmark->setOutputRootFileName(&outputRootFileName_);
}


void TauBenchmarkAnalyzer::endJob()
{
  benchmark->createPlots();
}

void TauBenchmarkAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  genJets_.clear();
  caloJets_.clear();
  pfJets_.clear();
 
  iEvent.getByLabel(caloJetsLabel_, caloJets_);
  iEvent.getByLabel(pfJetsLabel_, pfJets_);
  iEvent.getByLabel(genJetsLabel_, genJets_);
  iEvent.getByLabel("source", hepMC_);
  cout<<"Size of CaloJets: "<<caloJets_->size()<<endl;
  cout<<"Size of PFJets: "<<pfJets_->size()<<endl;
  cout<<"Size of genJets: "<<genJets_->size()<<endl;
  //  cout<<"Size of hepMC: "<<hepMC_->size()<<endl;
  
  
  
  
  benchmark->setCaloJets(caloJets_);
  benchmark->setPfJets(pfJets_);
  benchmark->setGenJets(genJets_);
  benchmark->setHepMC(hepMC_);
  benchmark->doBenchmark();

  
}











