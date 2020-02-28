// -*- C++ -*-
//
// Package:     HLTHiggsValidator
// Class:       HLTHiggsValidator
//

//
// Jordi Duarte Campderros (based on the Jason Slaunwhite
// and Jeff Klukas coded from the HLTriggerOffline/Muon package
//
//
//

// system include files
//#include<memory>

//#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTriggerOffline/Higgs/interface/HLTHiggsValidator.h"
#include "HLTriggerOffline/Higgs/src/EVTColContainer.cc"

//////// Class Methods ///////////////////////////////////////////////////////
// Constructor
HLTHiggsValidator::HLTHiggsValidator(const edm::ParameterSet& pset)
    : _pset(pset), _analysisnames(pset.getParameter<std::vector<std::string> >("analyses")), _collections(nullptr) {
  _collections = new EVTColContainer;

  //pass consumes list to the helper classes
  for (size_t i = 0; i < _analysisnames.size(); ++i) {
    HLTHiggsSubAnalysis analyzer(_pset, _analysisnames.at(i), consumesCollector());
    _analyzers.push_back(analyzer);
  }
}

HLTHiggsValidator::~HLTHiggsValidator() {
  if (_collections != nullptr) {
    delete _collections;
    _collections = nullptr;
  }
}

void HLTHiggsValidator::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  // Call the Plotter beginRun (which stores the triggers paths..:)
  for (std::vector<HLTHiggsSubAnalysis>::iterator iter = _analyzers.begin(); iter != _analyzers.end(); ++iter) {
    iter->beginRun(iRun, iSetup);
  }
}

void HLTHiggsValidator::bookHistograms(DQMStore::IBooker& ibooker,
                                       const edm::Run& iRun,
                                       const edm::EventSetup& iSetup) {
  // Call the Plotter bookHistograms (which stores the triggers paths..:)
  for (std::vector<HLTHiggsSubAnalysis>::iterator iter = _analyzers.begin(); iter != _analyzers.end(); ++iter) {
    iter->bookHistograms(ibooker);
  }
}

void HLTHiggsValidator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Initialize the event collections
  this->_collections->reset();

  for (std::vector<HLTHiggsSubAnalysis>::iterator iter = _analyzers.begin(); iter != _analyzers.end(); ++iter) {
    iter->analyze(iEvent, iSetup, this->_collections);
  }
}

//define this as a plug-in
//DEFINE_FWK_MODULE(HLTHiggsValidator);
