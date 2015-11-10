#include "FWCore/Utilities/interface/BranchType.h"

#include "HeavyIonsAnalysis/EventAnalysis/interface/RunAnalyzer.h"

RunAnalyzer::RunAnalyzer(const edm::ParameterSet& iConfig):
  genRunInfoToken_ (consumes<GenRunInfoProduct, edm::BranchType::InRun>(edm::InputTag("generator")))
{
    // Empty
}

RunAnalyzer::~RunAnalyzer() {
    // Empty
}

void RunAnalyzer::endRun(const edm::Run& run, const edm::EventSetup& iSetup) {
   
    edm::Handle<GenRunInfoProduct> genInfo;
    if (run.getByToken(genRunInfoToken_, genInfo)) {
        fXsec = genInfo->crossSection();
    }

    fTree->Fill();
}

// ------ method called once each job just before starting event loop  -------
void RunAnalyzer::beginJob()
{
  fTree = fs_->make<TTree>("run", "");
  fTree->Branch("xsec",&fXsec,"xsec/F");

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RunAnalyzer);
