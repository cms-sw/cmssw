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
L2MuonIsolationAnalyzer::L2MuonIsolationAnalyzer(const ParameterSet& pset) :
  theIsolationLabel(pset.getUntrackedParameter<edm::InputTag>("IsolationCollectionLabel")),
  theConeCases(pset.getParameter<std::vector<double> > ("ConeCases")),
  theEtMin(pset.getParameter<double> ("EtMin")),
  theEtMax(pset.getParameter<double> ("EtMax")),
  theEtBins(pset.getParameter<unsigned int> ("EtBins")),
  theCuts(pset.getParameter<std::vector<double> > ("EtaBounds"),
          pset.getParameter<std::vector<double> > ("ConeSizes"),
          pset.getParameter<std::vector<double> > ("Thresholds")),
  theRootFileName(pset.getUntrackedParameter<string>("rootFileName")),
  theTxtFileName(pset.getUntrackedParameter<string>("txtFileName"))
{
  LogDebug("Muon|RecoMuon|L2MuonIsolationTest")<<" L2MuonIsolationTest constructor called";


}

/// Destructor
L2MuonIsolationAnalyzer::~L2MuonIsolationAnalyzer(){
}

void L2MuonIsolationAnalyzer::beginJob(){
  // Create output files
  theTxtFile = fopen(theTxtFileName.data(), "w");

  theRootFile = TFile::Open(theRootFileName.c_str(), "RECREATE");
  theRootFile->cd();

  numberOfEvents = 0;
  numberOfMuons = 0;

  hEtSum = new TH1F("etSum","Sum E_{T}^{weighted} (GeV)",100,0,50);
  hEffVsCone = new TH1F("effVsCone","Efficiency(%) vs Cone Set"
      , theConeCases.size(), 0.5, theConeCases.size()+0.5);
  hEffVsEt = new TH1F("effVsEt","Efficiency(%) vs E_{T}^{weighted} (GeV)"
      , theEtBins, theEtMin, theEtMax);

  hEffVsEtArray.clear();
  char chnam[256];
  char chtit[256];
  for (unsigned int j=0; j<theConeCases.size() ; j++) {
      for (unsigned int k=0; k<theCuts.size() ; k++) {
            snprintf(chnam,sizeof(chnam),"effVsEt-%.2d-%.2d", j, k);
            snprintf(chtit,sizeof(chtit),"Eff(%%) vs E_{T}(GeV), cone size %.4f, eta in [%.4f,%.4f]"
               , theConeCases[j], theCuts[k].etaRange.min(), theCuts[k].etaRange.max());
            hEffVsEtArray.push_back(new TH1F(chnam, chtit
                  , theEtBins,theEtMin,theEtMax));
      }
  }

}

void L2MuonIsolationAnalyzer::endJob(){
  Puts("L2MuonIsolationAnalyzer>>> FINAL PRINTOUTS -> BEGIN");
  Puts("L2MuonIsolationAnalyzer>>> Number of analyzed events= %d", numberOfEvents);
  Puts("L2MuonIsolationAnalyzer>>> Number of analyzed muons= %d", numberOfMuons);
    
  // Write the histos to file
  theRootFile->cd();

  hEtSum->Write();

  unsigned int overflow_bin;
  float norm;

  overflow_bin = hEffVsCone->GetNbinsX()+1;
  norm = hEffVsCone->GetBinContent(overflow_bin);
  if (norm<=0.) norm = 1.;
  Puts("EffVsCone Normalization= %f", norm);
  for (unsigned int k=1; k<overflow_bin; k++) {
            float aux = hEffVsCone->GetBinContent(k);
            float eff = 100*aux/norm;
            hEffVsCone->SetBinContent(k,eff);
            Puts("\tEfficiency in cone index= %d (size=%f): %f"
                        , k, theConeCases[k-1], eff);
  }
  hEffVsCone->Write();

  Puts("");
  overflow_bin = hEffVsEt->GetNbinsX()+1;
  norm = hEffVsEt->GetBinContent(overflow_bin);
  if (norm<=0.) norm = 1.;
  Puts("EffVsEt Normalization= %f", norm);
  for (unsigned int k=1; k<overflow_bin; k++) {
            float aux = hEffVsEt->GetBinContent(k);
            float eff = 100*aux/norm;
            hEffVsEt->SetBinContent(k,eff);
            float et = theEtMin + (k-0.5)/theEtBins*(theEtMax-theEtMin);
            Puts("\tEfficiency for Et threshold cut %f: %f", et, eff);
  }
  hEffVsEt->Write();

  for (unsigned int j=0; j<hEffVsEtArray.size(); j++) {
      overflow_bin = hEffVsEtArray[j]->GetNbinsX()+1;
      norm = hEffVsEtArray[j]->GetBinContent(overflow_bin);
      if (norm<=0.) norm = 1.;
      Puts("EffVsEt[%d] Normalization= %f", j, norm);
      fprintf(theTxtFile, "%s\n", hEffVsEtArray[j]->GetTitle());
      fprintf(theTxtFile, "%s\n", "ET threshold(GeV), efficiency(%): ");
      for (unsigned int k=1; k<overflow_bin; k++) {
            float aux = hEffVsEtArray[j]->GetBinContent(k);
            float eff = 100*aux/norm;
            hEffVsEtArray[j]->SetBinContent(k,eff);
            float etthr = hEffVsEtArray[j]->GetXaxis()->GetBinCenter(k);
            fprintf(theTxtFile, "%6.2f %8.3f\n", etthr, eff);
      }
      hEffVsEtArray[j]->Write();
  }

  theRootFile->Close();
  fclose(theTxtFile);

  Puts("L2MuonIsolationAnalyzer>>> FINAL PRINTOUTS -> END");
}

void L2MuonIsolationAnalyzer::analyze(const Event & event, const EventSetup& eventSetup){
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
      //Puts("L2MuonIsolationAnalyzer>>> Track pt: %f, Deposit eta%f", tk->pt(), dep.eta());
      
      muonisolation::Cuts::CutSpec cuts_here = theCuts(tk->eta());
      double conesize = cuts_here.conesize;
      float etsum = dep.depositWithin(conesize);
      hEtSum->Fill(etsum);
      
      hEffVsCone->Fill(theConeCases.size()+999.);
      for (unsigned int j=0; j<theConeCases.size(); j++) {
	if (dep.depositWithin(theConeCases[j])<cuts_here.threshold) 
	  hEffVsCone->Fill(j+1.0);
      }
      
      hEffVsEt->Fill(theEtMax+999.);
      for (unsigned int j=0; j<theEtBins; j++) {
	float etthr = theEtMin + (j+0.5)/theEtBins*(theEtMax-theEtMin);
	if (etsum<etthr) hEffVsEt->Fill(etthr);
      }
      
      for (unsigned int j=0; j<theConeCases.size() ; j++) {
	float etsum = dep.depositWithin(theConeCases[j]);
	for (unsigned int k=0; k<theCuts.size() ; k++) {
	  unsigned int n = theCuts.size()*j + k;
	  if (fabs(tk->eta())<theCuts[k].etaRange.min()) continue;
	  if (fabs(tk->eta())>theCuts[k].etaRange.max()) continue;
	  hEffVsEtArray[n]->Fill(theEtMax+999.);
	  for (unsigned int l=0; l<theEtBins; l++) {
	    float etthr = theEtMin + (l+0.5)/theEtBins*(theEtMax-theEtMin);
	    if (etsum<etthr) hEffVsEtArray[n]->Fill(etthr);
	  }
	}
      }
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
      va_end(ap);
      LogVerbatim("") << chout;
      //LogError("") << chout;
}

DEFINE_FWK_MODULE(L2MuonIsolationAnalyzer);
