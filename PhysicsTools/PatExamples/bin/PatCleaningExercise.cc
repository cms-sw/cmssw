#include <vector>

#include "TH1.h"
#include "TFile.h"
#include <TROOT.h>
#include <TSystem.h>

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "FWCore/ParameterSetReader/interface/ProcessDescImpl.h"

//using namespace std;
//using namespace reco;
//using namespace pat;

int main(int argc, char* argv[]) {
  // ----------------------------------------------------------------------
  // First Part:
  //
  //  * enable FWLite
  //  * book the histograms of interest
  //  * open the input file
  // ----------------------------------------------------------------------

  // load framework libraries
  gSystem->Load("libFWCoreFWLite");
  FWLiteEnabler::enable();

  // only allow one argument for this simple example which should be the
  // the python cfg file
  if (argc < 2) {
    std::cout << "Usage : " << argv[0] << " [parameters.py]" << std::endl;
    return 0;
  }

  // get the python configuration
  ProcessDescImpl builder(argv[1], true);
  const edm::ParameterSet& fwliteParameters =
      builder.processDesc()->getProcessPSet()->getParameter<edm::ParameterSet>("FWLiteParams");

  // now get each parameter
  std::string input_(fwliteParameters.getParameter<std::string>("inputFile"));
  std::string output_(fwliteParameters.getParameter<std::string>("outputFile"));
  std::string overlaps_(fwliteParameters.getParameter<std::string>("overlaps"));
  edm::InputTag jets_(fwliteParameters.getParameter<edm::InputTag>("jets"));

  // book a set of histograms
  fwlite::TFileService fs = fwlite::TFileService(output_);
  TFileDirectory theDir = fs.mkdir("analyzePatCleaning");
  TH1F* emfAllJets_ = theDir.make<TH1F>("emfAllJets", "f_{emf}(All Jets)", 20, 0., 1.);
  TH1F* emfCleanJets_ = theDir.make<TH1F>("emfCleanJets", "f_{emf}(Clean Jets)", 20, 0., 1.);
  TH1F* emfOverlapJets_ = theDir.make<TH1F>("emfOverlapJets", "f_{emf}(Overlap Jets)", 20, 0., 1.);
  TH1F* deltaRElecJet_ = theDir.make<TH1F>("deltaRElecJet", "#DeltaR (elec, jet)", 10, 0., 0.5);
  TH1F* elecOverJet_ = theDir.make<TH1F>("elecOverJet", "E_{elec}/E_{jet}", 100, 0., 2.);
  TH1F* nOverlaps_ = theDir.make<TH1F>("nOverlaps", "Number of overlaps", 5, 0., 5.);

  // open input file (can be located on castor)
  TFile* inFile = TFile::Open(input_.c_str());

  // ----------------------------------------------------------------------
  // Second Part:
  //
  //  * loop the events in the input file
  //  * receive the collections of interest via fwlite::Handle
  //  * fill the histograms
  //  * after the loop close the input file
  // ----------------------------------------------------------------------

  // loop the events
  unsigned int iEvent = 0;
  fwlite::Event ev(inFile);
  for (ev.toBegin(); !ev.atEnd(); ++ev, ++iEvent) {
    edm::EventBase const& event = ev;

    // break loop after end of file is reached
    // or after 1000 events have been processed
    if (iEvent == 1000)
      break;

    // simple event counter
    if (iEvent > 0 && iEvent % 1 == 0) {
      std::cout << "  processing event: " << iEvent << std::endl;
    }

    // handle to jet collection
    edm::Handle<std::vector<pat::Jet> > jets;
    event.getByLabel(jets_, jets);

    // loop over the jets in the event
    for (std::vector<pat::Jet>::const_iterator jet = jets->begin(); jet != jets->end(); jet++) {
      if (jet->pt() > 20 && jet == jets->begin()) {
        emfAllJets_->Fill(jet->emEnergyFraction());
        if (!jet->hasOverlaps(overlaps_)) {
          emfCleanJets_->Fill(jet->emEnergyFraction());
        } else {
          //get all overlaps
          const reco::CandidatePtrVector overlaps = jet->overlaps(overlaps_);
          nOverlaps_->Fill(overlaps.size());
          emfOverlapJets_->Fill(jet->emEnergyFraction());
          //loop over the overlaps
          for (reco::CandidatePtrVector::const_iterator overlap = overlaps.begin(); overlap != overlaps.end();
               overlap++) {
            float deltaR = reco::deltaR((*overlap)->eta(), (*overlap)->phi(), jet->eta(), jet->phi());
            deltaRElecJet_->Fill(deltaR);
            elecOverJet_->Fill((*overlap)->energy() / jet->energy());
          }
        }
      }
    }
  }
  inFile->Close();
  return 0;
}
