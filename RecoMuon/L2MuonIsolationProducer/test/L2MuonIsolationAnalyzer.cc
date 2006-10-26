/** \class L2MuonIsolationAnalyzer
 *  Analyzer of HLT L2 muon isolation performance
 *
 *  \author J. Alcaraz
 */

#include "RecoMuon/L2MuonIsolationProducer/test/L2MuonIsolationAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TFile.h"
#include "TH1F.h"

using namespace std;
using namespace edm;
using namespace reco;

/// Constructor
L2MuonIsolationAnalyzer::L2MuonIsolationAnalyzer(const ParameterSet& pset) :
  theIsolationLabel(pset.getUntrackedParameter<string>("IsolationCollectionLabel")),
  theConeCases(pset.getParameter<std::vector<double> > ("ConeCases")),
  theEtCases(pset.getParameter<std::vector<double> > ("EtCases")),
  theCuts(pset.getParameter<std::vector<double> > ("EtaBounds"),
          pset.getParameter<std::vector<double> > ("ConeSizes"),
          pset.getParameter<std::vector<double> > ("Thresholds")),
  theRootFileName(pset.getUntrackedParameter<string>("rootFileName"))
{
  LogDebug("Muon|RecoMuon|L2MuonIsolationTest")<<" L2MuonIsolationTest constructor called";


}

/// Destructor
L2MuonIsolationAnalyzer::~L2MuonIsolationAnalyzer(){
}

void L2MuonIsolationAnalyzer::beginJob(const EventSetup& eventSetup){
  // Create the root file
  theRootFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theRootFile->cd();

  numberOfEvents = 0;
  numberOfMuons = 0;
  numberOfSelectedMuonsVsCone.clear();
  numberOfSelectedMuonsVsCone.resize(theConeCases.size(),0);
  numberOfSelectedMuonsVsEt.clear();
  numberOfSelectedMuonsVsEt.resize(theEtCases.size(),0);

  hEtSum = new TH1F("etSum","Sum E_{T}^{weighted} (GeV)",100,0,50);
  hEffVsCone = new TH1F("effVsCone","Efficiency(%) vs Cone Set"
      , theConeCases.size(), 0.5, theConeCases.size()+0.5);
  hEffVsEt = new TH1F("effVsEt","Efficiency(%) vs E_{T}^{weighted} (GeV)"
      , theEtCases.size(), theEtCases[0]-0.25
      , theEtCases[theEtCases.size()-1]+0.25);

}

void L2MuonIsolationAnalyzer::endJob(){
  Puts("L2MuonIsolationAnalyzer>>> FINAL PRINTOUTS -> BEGIN");
  Puts("L2MuonIsolationAnalyzer>>> Number of analyzed events= %d", numberOfEvents);
  Puts("L2MuonIsolationAnalyzer>>> Number of analyzed muons= %d", numberOfMuons);
  Puts("");
  for (unsigned int j=0; j<theConeCases.size(); j++) {
      float eff = numberOfSelectedMuonsVsCone[j]*100./numberOfMuons;
      Puts("\tEfficiency in cone index= %d (size=%f): %f"
                  , j+1, theConeCases[j], eff);
      hEffVsCone->Fill(j+1., eff);
  }
  Puts("");
  for (unsigned int j=0; j<theEtCases.size(); j++) {
      float eff = numberOfSelectedMuonsVsEt[j]*100./numberOfMuons;
      Puts("\tEfficiency for Et threshold cut %f: %f"
                  , theEtCases[j], eff);
      hEffVsEt->Fill(theEtCases[j], eff);
  }
  Puts("L2MuonIsolationAnalyzer>>> FINAL PRINTOUTS -> END");
    
  // Write the histos to file
  theRootFile->cd();

  hEtSum->Write();
  hEffVsCone->Write();
  hEffVsEt->Write();

  theRootFile->Close();
}
 

void L2MuonIsolationAnalyzer::analyze(const Event & event, const EventSetup& eventSetup){
  
  numberOfEvents++;
  
  // Get the isolation information from the event
  Handle<MuIsoDepositAssociationMap> depMap;
  event.getByLabel(theIsolationLabel, depMap);

  MuIsoDepositAssociationMap::const_iterator depPair;
  for (depPair=depMap->begin(); depPair!=depMap->end(); ++depPair) {
      numberOfMuons++;
      const TrackRef& tk = depPair->key;
      const MuIsoDeposit& dep = depPair->val;
      //Puts("L2MuonIsolationAnalyzer>>> Track pt: %f, Deposit eta%f", tk->pt(), dep.eta());

      muonisolation::Cuts::CutSpec cuts_here = theCuts(tk->eta());
      double conesize = cuts_here.conesize;
      float etsum = dep.depositWithin(conesize);
      hEtSum->Fill(etsum);
      for (unsigned int j=0; j<theConeCases.size(); j++) {
            if (dep.depositWithin(theConeCases[j])<cuts_here.threshold) numberOfSelectedMuonsVsCone[j]++;
      }
      for (unsigned int j=0; j<theEtCases.size(); j++) {
            if (etsum<theEtCases[j]) numberOfSelectedMuonsVsEt[j]++;
      }
  }

  Puts("L2MuonIsolationAnalyzer>>> Run: %ld, Event %ld, number of muons %d"
                 , event.id().run(), event.id().event(), depMap->size());
}

void L2MuonIsolationAnalyzer::Puts(const char* va_(fmt), ...) {
      // Do not write more than 256 characters
      const unsigned int bufsize = 256; 
      char chout[bufsize] = "";
      va_list ap;
      va_start(ap, va_(fmt));
      vsnprintf(chout, bufsize, va_(fmt), ap);
      LogVerbatim("") << chout;
      va_end(ap);
}

DEFINE_FWK_MODULE(L2MuonIsolationAnalyzer);
