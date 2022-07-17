// -*- C++ -*-
//
// Package:    TreeSplitter
// Class:      TreeSplitter
//
/**\class TreeSplitter TreeSplitter.cc MuonAnalysis/MomentumScaleCalibration/plugins/TreeSplitter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Marco De Mattia
//         Created:  Thu Sep 11 12:16:00 CEST 2008
//

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <MuonAnalysis/MomentumScaleCalibration/interface/RootTreeHandler.h>

class TreeSplitter : public edm::one::EDAnalyzer<> {
public:
  explicit TreeSplitter(const edm::ParameterSet&);
  ~TreeSplitter() override;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override{};
  void endJob() override;

  TString treeFileName_;
  TString outputFileName_;
  int32_t maxEvents_;
  uint32_t subSampleFirstEvent_;
  uint32_t subSampleMaxEvents_;
};

TreeSplitter::TreeSplitter(const edm::ParameterSet& iConfig)
    : treeFileName_(iConfig.getParameter<std::string>("InputFileName")),
      outputFileName_(iConfig.getParameter<std::string>("OutputFileName")),
      maxEvents_(iConfig.getParameter<int32_t>("MaxEvents")),
      subSampleFirstEvent_(iConfig.getParameter<uint32_t>("SubSampleFirstEvent")),
      subSampleMaxEvents_(iConfig.getParameter<uint32_t>("SubSampleMaxEvents")) {}

TreeSplitter::~TreeSplitter() {}

void TreeSplitter::endJob() {
  std::cout << "Reading muon pairs from Root Tree in " << treeFileName_ << std::endl;
  RootTreeHandler rootTreeHandler;

  typedef std::vector<std::pair<lorentzVector, lorentzVector> > MuonPairVector;
  // MuonPairVector savedPair;
  std::vector<MuonPair> savedPair;
  rootTreeHandler.readTree(maxEvents_, treeFileName_, &savedPair, 0);
  // rootTreeHandler.readTree(maxEvents, inputRootTreeFileName_, &savedPair, &(MuScleFitUtils::genPair));

  // Loop on all the pairs
  std::vector<MuonPair> newSavedPair;
  // MuonPairVector newSavedPair;
  unsigned int i = 0;
  // MuonPairVector::iterator it = savedPair.begin();
  std::vector<MuonPair>::iterator it = savedPair.begin();
  std::cout << "Starting loop on " << savedPair.size() << " muons" << std::endl;
  uint32_t lastEvent = subSampleFirstEvent_ + subSampleMaxEvents_;
  for (; it != savedPair.end(); ++it, ++i) {
    // Save only events in the selected range
    if (i >= subSampleFirstEvent_ && i < lastEvent) {
      newSavedPair.push_back(*it);
    }
  }
  rootTreeHandler.writeTree(outputFileName_, &newSavedPair, 0);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TreeSplitter);
