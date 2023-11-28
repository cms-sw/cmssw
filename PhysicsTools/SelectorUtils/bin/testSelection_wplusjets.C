#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "DataFormats/FWLite/interface/MultiChainEvent.h"
#include "PhysicsTools/SelectorUtils/interface/WPlusJetsEventSelector.h"
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "FWCore/ParameterSetReader/interface/ProcessDescImpl.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"

#include "Math/GenVector/PxPyPzM4D.h"

#include <iostream>
#include <cmath>  //necessary for absolute function fabs()

//Root includes
#include "TROOT.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TSystem.h"

using namespace std;

int main(int argc, char** argv) {
  // load framework libraries
  gSystem->Load("libFWCoreFWLite");
  FWLiteEnabler::enable();

  if (argc < 2) {
    std::cout << "Usage : " << argv[0] << " [parameters.py]" << std::endl;
    return 0;
  }

  // Get the python configuration
  ProcessDescImpl builder(argv[1], true);
  edm::ParameterSet const& shyftParameters =
      builder.processDesc()->getProcessPSet()->getParameter<edm::ParameterSet>("wplusjetsAnalysis");
  edm::ParameterSet const& inputs = builder.processDesc()->getProcessPSet()->getParameter<edm::ParameterSet>("inputs");
  edm::ParameterSet const& outputs =
      builder.processDesc()->getProcessPSet()->getParameter<edm::ParameterSet>("outputs");

  // book a set of histograms
  fwlite::TFileService fs = fwlite::TFileService(outputs.getParameter<std::string>("outputName"));
  TFileDirectory theDir = fs.mkdir("histos");

  // This object 'event' is used both to get all information from the
  // event as well as to store histograms, etc.
  fwlite::ChainEvent ev(inputs.getParameter<std::vector<std::string> >("fileNames"));

  //cout << "Making event selector" << endl;
  WPlusJetsEventSelector wPlusJets(shyftParameters);
  pat::strbitset ret = wPlusJets.getBitTemplate();

  //loop through each event
  for (ev.toBegin(); !ev.atEnd(); ++ev) {
    ret.set(false);

    wPlusJets(ev, ret);

    std::vector<reco::ShallowClonePtrCandidate> const& electrons = wPlusJets.selectedElectrons();
    std::vector<reco::ShallowClonePtrCandidate> const& muons = wPlusJets.selectedMuons();
    std::vector<reco::ShallowClonePtrCandidate> const& jets = wPlusJets.cleanedJets();
    std::vector<reco::ShallowClonePtrCandidate> const& jetsBeforeClean = wPlusJets.selectedJets();

    string bit_;

    bit_ = "Trigger";
    bool passTrigger = ret[bit_];
    bit_ = "== 1 Lepton";
    bool passOneLepton = ret[bit_];
    bit_ = "= 0 Jets";
    // bool jet0 = ret[bit_];
    bit_ = ">=1 Jets";
    bool jet1 = ret[bit_];
    bit_ = ">=2 Jets";
    bool jet2 = ret[bit_];
    bit_ = ">=3 Jets";
    bool jet3 = ret[bit_];
    bit_ = ">=4 Jets";
    bool jet4 = ret[bit_];
    bit_ = ">=5 Jets";
    bool jet5 = ret[bit_];

    bool anyJets = jet1 || jet2 || jet3 || jet4 || jet5;

    if (anyJets && passOneLepton && passTrigger) {
      cout << "Nele = " << electrons.size() << ", Nmuo = " << muons.size() << ", Njets_all = " << jets.size()
           << ", Njets_clean = " << jetsBeforeClean.size() << endl;
    }

  }  //end event loop

  cout << "Printing" << endl;
  wPlusJets.print(std::cout);
  cout << "We're done!" << endl;

  return 0;
}
