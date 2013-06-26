/** \class L3MuonIsolationAnalyzer
 *  Analyzer of HLT L3 muon isolation performance
 *
 *  \author J. Alcaraz
 */

#include "RecoMuon/L3MuonIsolationProducer/test/L3MuonIsolationAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TFile.h"
#include "TH1F.h"

#include "Varargs.h"

using namespace std;
using namespace edm;
using namespace reco;

/// Constructor
L3MuonIsolationAnalyzer::L3MuonIsolationAnalyzer(const ParameterSet& pset) :
  theIsolationLabel(pset.getUntrackedParameter<edm::InputTag>("IsolationCollectionLabel")),
  theConeCases(pset.getParameter<std::vector<double> > ("ConeCases")),
  thePtMin(pset.getParameter<double> ("PtMin")),
  thePtMax(pset.getParameter<double> ("PtMax")),
  thePtBins(pset.getParameter<unsigned int> ("PtBins")),
  theCuts(pset.getParameter<std::vector<double> > ("EtaBounds"),
          pset.getParameter<std::vector<double> > ("ConeSizes"),
          pset.getParameter<std::vector<double> > ("Thresholds")),
  theRootFileName(pset.getUntrackedParameter<string>("rootFileName")),
  theTxtFileName(pset.getUntrackedParameter<string>("txtFileName"))
{
  LogDebug("Muon|RecoMuon|L3MuonIsolationTest")<<" L3MuonIsolationTest constructor called";


}

/// Destructor
L3MuonIsolationAnalyzer::~L3MuonIsolationAnalyzer(){
}

void L3MuonIsolationAnalyzer::beginJob(){
  // Create output files
  theTxtFile = fopen(theTxtFileName.data(), "w");

  theRootFile = TFile::Open(theRootFileName.c_str(), "RECREATE");
  theRootFile->cd();

  numberOfEvents = 0;
  numberOfMuons = 0;

  hPtSum = new TH1F("ptSum","Sum E_{T}^{weighted} (GeV)",100,0,50);
  hEffVsCone = new TH1F("effVsCone","Efficiency(%) vs Cone Set"
      , theConeCases.size(), 0.5, theConeCases.size()+0.5);
  hEffVsPt = new TH1F("effVsPt","Efficiency(%) vs E_{T}^{weighted} (GeV)"
      , thePtBins, thePtMin, thePtMax);

  hEffVsPtArray.clear();
  char chnam[256];
  char chtit[256];
  for (unsigned int j=0; j<theConeCases.size() ; j++) {
      for (unsigned int k=0; k<theCuts.size() ; k++) {
            snprintf(chnam,sizeof(chnam),"effVsPt-%.2d-%.2d", j, k);
            snprintf(chtit,sizeof(chtit),"Eff(%%) vs P_{T}(GeV), cone size %.4f, eta in [%.4f,%.4f]"
               , theConeCases[j], theCuts[k].etaRange.min(), theCuts[k].etaRange.max());
            hEffVsPtArray.push_back(new TH1F(chnam, chtit
                  , thePtBins,thePtMin,thePtMax));
      }
  }

}

void L3MuonIsolationAnalyzer::endJob(){
  Puts("L3MuonIsolationAnalyzer>>> FINAL PRINTOUTS -> BEGIN");
  Puts("L3MuonIsolationAnalyzer>>> Number of analyzed events= %d", numberOfEvents);
  Puts("L3MuonIsolationAnalyzer>>> Number of analyzed muons= %d", numberOfMuons);
    
  // Write the histos to file
  theRootFile->cd();

  hPtSum->Write();

  unsigned int overflow_bin;
  float norm;

  overflow_bin = hEffVsCone->GetNbinsX()+1;
  norm = hEffVsCone->GetBinContent(overflow_bin);
  if (norm<=0.) norm = 1.;
  Puts("");
  for (unsigned int k=1; k<overflow_bin; k++) {
            float aux = hEffVsCone->GetBinContent(k);
            float eff = 100*aux/norm;
            hEffVsCone->SetBinContent(k,eff);
            Puts("\tEfficiency in cone index= %d (size=%f): %f"
                        , k, theConeCases[k-1], eff);
  }
  hEffVsCone->Write();

  Puts("");
  overflow_bin = hEffVsPt->GetNbinsX()+1;
  norm = hEffVsPt->GetBinContent(overflow_bin);
  if (norm<=0.) norm = 1.;
  for (unsigned int k=1; k<overflow_bin; k++) {
            float aux = hEffVsPt->GetBinContent(k);
            float eff = 100*aux/norm;
            hEffVsPt->SetBinContent(k,eff);
            float pt = thePtMin + (k-0.5)/thePtBins*(thePtMax-thePtMin);
            Puts("\tEfficiency for Pt threshold cut %f: %f", pt, eff);
  }
  hEffVsPt->Write();

  for (unsigned int j=0; j<hEffVsPtArray.size(); j++) {
      overflow_bin = hEffVsPtArray[j]->GetNbinsX()+1;
      norm = hEffVsPtArray[j]->GetBinContent(overflow_bin);
      if (norm<=0.) norm = 1.;
      fprintf(theTxtFile, "%s\n", hEffVsPtArray[j]->GetTitle());
      fprintf(theTxtFile, "%s\n", "PT threshold(GeV), efficiency(%): ");
      for (unsigned int k=1; k<overflow_bin; k++) {
            float aux = hEffVsPtArray[j]->GetBinContent(k);
            float eff = 100*aux/norm;
            hEffVsPtArray[j]->SetBinContent(k,eff);
            float ptthr = hEffVsPtArray[j]->GetXaxis()->GetBinCenter(k);
            fprintf(theTxtFile, "%6.2f %8.3f\n", ptthr, eff);
      }
      hEffVsPtArray[j]->Write();
  }

  theRootFile->Close();
  fclose(theTxtFile);

  Puts("L3MuonIsolationAnalyzer>>> FINAL PRINTOUTS -> END");
}
 

void L3MuonIsolationAnalyzer::analyze(const Event & event, const EventSetup& eventSetup){
  static const string metname = "L3MuonIsolation";
  
  numberOfEvents++;
  
  // Get the isolation information from the event
  Handle<reco::IsoDepositMap> depMap;
  event.getByLabel(theIsolationLabel, depMap);

  //! How do I know it's made for tracks? Just guessing
  typedef edm::View<reco::Track>  tracks_type;
  Handle<tracks_type > tracksH;

  typedef reco::IsoDepositMap::const_iterator depmap_iterator;
  depmap_iterator depMI = depMap->begin();
  depmap_iterator depMEnd = depMap->end();

  for (; depMI != depMEnd; ++depMI) {
    if (depMI.id() != tracksH.id()){
      LogTrace(metname)<<"Getting tracks with id "<<depMI.id();
      event.get(depMI.id(), tracksH);
    }
    
    typedef reco::IsoDepositMap::container::const_iterator dep_iterator;
    dep_iterator depI = depMI.begin();
    dep_iterator depEnd = depMI.end();

    typedef tracks_type::const_iterator tk_iterator;
    tk_iterator tkI = tracksH->begin();
    tk_iterator tkEnd = tracksH->end();

    for (;tkI != tkEnd && depI != depEnd; ++tkI, ++depI){    
      
      numberOfMuons++;
      tracks_type::const_pointer tk = &*tkI;
      const IsoDeposit& dep = *depI;
      
      muonisolation::Cuts::CutSpec cuts_here = theCuts(tk->eta());
      double conesize = cuts_here.conesize;
      float ptsum = dep.depositWithin(conesize);
      hPtSum->Fill(ptsum);
      
      hEffVsCone->Fill(theConeCases.size()+999.);
      for (unsigned int j=0; j<theConeCases.size(); j++) {
	if (dep.depositWithin(theConeCases[j])<cuts_here.threshold) 
	  hEffVsCone->Fill(j+1.0);
      }
      
      hEffVsPt->Fill(thePtMax+999.);
      for (unsigned int j=0; j<thePtBins; j++) {
	float ptthr = thePtMin + (j+0.5)/thePtBins*(thePtMax-thePtMin);
	if (ptsum<ptthr) hEffVsPt->Fill(ptthr);
      }
      
      for (unsigned int j=0; j<theConeCases.size() ; j++) {
	float ptsum = dep.depositWithin(theConeCases[j]);
	for (unsigned int k=0; k<theCuts.size() ; k++) {
	  unsigned int n = theCuts.size()*j + k;
	  if (fabs(tk->eta())<theCuts[k].etaRange.min()) continue;
	  if (fabs(tk->eta())>theCuts[k].etaRange.max()) continue;
	  hEffVsPtArray[n]->Fill(thePtMax+999.);
	  for (unsigned int l=0; l<thePtBins; l++) {
	    float ptthr = thePtMin + (l+0.5)/thePtBins*(thePtMax-thePtMin);
	    if (ptsum<ptthr) hEffVsPtArray[n]->Fill(ptthr);
	  }
	}
      }
    }

  }
  Puts("L3MuonIsolationAnalyzer>>> Run: %ld, Event %ld, number of muons %d"
                 , event.id().run(), event.id().event(), depMap->size());
}

void L3MuonIsolationAnalyzer::Puts(const char* va_(fmt), ...) {
      // Do not write more than 256 characters
      const unsigned int bufsize = 256; 
      char chout[bufsize] = "";
      va_list ap;
      va_start(ap, va_(fmt));
      vsnprintf(chout, bufsize, va_(fmt), ap);
      va_end(ap);
      LogVerbatim("") << chout;
}

DEFINE_FWK_MODULE(L3MuonIsolationAnalyzer);
