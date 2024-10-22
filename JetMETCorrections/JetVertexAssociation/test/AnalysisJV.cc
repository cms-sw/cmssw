
#include <memory>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "JetMETCorrections/JetVertexAssociation/test/AnalysisJV.h"

using namespace edm;
using namespace std;
using namespace reco;

AnalysisJV::AnalysisJV(const edm::ParameterSet& pset)
    : fOutputFileName(pset.getUntrackedParameter<string>("HistOutFile", std::string("jv_analysis.root"))),
      fResult1Token(consumes<ResultCollection1>(edm::InputTag("jetvertex", "Var"))),
      fResult2Token(consumes<ResultCollection2>(edm::InputTag("jetvertex", "JetType"))),
      fCaloJetsToken(consumes<CaloJetCollection>(edm::InputTag("iterativeCone5CaloJets"))) {}

AnalysisJV::~AnalysisJV() {}

void AnalysisJV::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  cout << "----------------------------" << endl;

  Handle<ResultCollection1> JV_alpha;
  iEvent.getByToken(fResult1Token, JV_alpha);

  Handle<ResultCollection2> JV_jet_type;
  iEvent.getByToken(fResult2Token, JV_jet_type);

  Handle<CaloJetCollection> CaloIconeJetsHandle;
  iEvent.getByToken(fCaloJetsToken, CaloIconeJetsHandle);

  if (CaloIconeJetsHandle->size()) {
    ResultCollection1::const_iterator it_jv1 = JV_alpha->begin();
    ResultCollection2::const_iterator it_jv2 = JV_jet_type->begin();
    for (CaloJetCollection::const_iterator it = CaloIconeJetsHandle->begin(); it != CaloIconeJetsHandle->end(); it++) {
      if (*it_jv2)
        cout << "Jet: Et = " << it->pt() << " - true jet" << endl;
      else
        cout << "Jet: Et = " << it->pt() << " - 'fake' jet" << endl;

      fHistAlpha->Fill(*it_jv1);
      it_jv1++;
      it_jv2++;
    }
  }
}

void AnalysisJV::beginJob() {
  fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
  fHistAlpha = new TH1D("HistAlpha", "", 30, 0., 1.5);
}

void AnalysisJV::endJob() {
  fOutputFile->Write();
  fOutputFile->Close();
  delete fOutputFile;

  return;
}

//define this as a plug-in
DEFINE_FWK_MODULE(AnalysisJV);
