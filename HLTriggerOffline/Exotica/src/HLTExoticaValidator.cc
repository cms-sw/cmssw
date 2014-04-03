// -*- C++ -*-
//
// Package:     HLTExoticaValidator
// Class:       HLTExoticaValidator
//

//
// Jordi Duarte Campderros (based on the Jason Slaunwhite
// and Jeff Klukas coded from the HLTriggerOffline/Muon package
//
//
//

// system include files

#include "HLTriggerOffline/Exotica/interface/HLTExoticaValidator.h"
#include "HLTriggerOffline/Exotica/src/EVTColContainer.cc"

//////// Class Methods ///////////////////////////////////////////////////////

// Constructor
HLTExoticaValidator::HLTExoticaValidator(const edm::ParameterSet& pset) :
    _pset(pset),
    _analysisnames(pset.getParameter<std::vector<std::string> >("analysis")),
    _collections(0)
{

    LogDebug("ExoticaValidation") << "In HLTExoticaValidator::constructor()";

    // Prepare the event collections to be used.
    _collections = new EVTColContainer;

    // Create a new subanalysis for each of the analysis names.
    // Notice that the constructor takes the full parameter set,
    // the analysis name and the consumesCollector() separately.
    for (size_t i = 0; i < _analysisnames.size() ; ++i) {
        HLTExoticaSubAnalysis analyzer(_pset, _analysisnames.at(i), consumesCollector());
        _analyzers.push_back(analyzer);
    }

}

HLTExoticaValidator::~HLTExoticaValidator()
{
    if (_collections != 0) {
        delete _collections;
        _collections = 0;
    }
}


// 2014-02-03 -- Thiago                                                      
// Due to the fact that the DQM has to be thread safe now, we have to do things differently:                                                         
// 1) Implement the bookHistograms method in this class
// 2) Split beginRun() into bookHistograms() and dqmBeginRun()                                                                                
// 3) Call subAnalysisBookHistos() for each subAnalysis from inside bookHistograms()
// *** IMPORTANT *** notice that now dqmBeginRun() runs before bookHistograms()!
void HLTExoticaValidator::dqmBeginRun(const edm::Run & iRun, const edm::EventSetup & iSetup)
{
    LogDebug("ExoticaValidation") << "In HLTExoticaValidator::dqmBeginRun()";

    // Call the Plotter beginRun (which stores the triggers paths..:)
    for (std::vector<HLTExoticaSubAnalysis>::iterator iter = _analyzers.begin();
         iter != _analyzers.end(); ++iter) {
        iter->beginRun(iRun, iSetup);
    }
}

void HLTExoticaValidator::bookHistograms(DQMStore::IBooker &iBooker, const edm::Run & iRun, const edm::EventSetup & iSetup)
{

    LogDebug("ExoticaValidation") << "In HLTExoticaValidator::bookHistograms()";

    // Loop over all sub-analyses and book histograms for all of them.
    // For this to work, I think we have to pass the iBooker to each of them.
    // I don't think we have any guarantee that this loop is executed sequentially,
    // but the booking with iBooker itself has such a guarantee.
    for (std::vector<HLTExoticaSubAnalysis>::iterator iter = _analyzers.begin();
         iter != _analyzers.end(); ++iter) {
        iter->subAnalysisBookHistos(iBooker, iRun, iSetup);
    }
}


void HLTExoticaValidator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    LogDebug("ExoticaValidation") << "In HLTExoticaValidator::analyze()";

    //static int eventNumber = 0;
    //eventNumber++;
    //LogDebug("ExoticaValidation") << "In HLTExoticaSubAnalysis::analyze,  "
    //                              << "Event: " << eventNumber;

    // Initialize the event collections
    this->_collections->reset();

    for (std::vector<HLTExoticaSubAnalysis>::iterator iter = _analyzers.begin();
         iter != _analyzers.end(); ++iter) {
        iter->analyze(iEvent, iSetup, this->_collections);
    }
}



void HLTExoticaValidator::beginJob()
{
    LogDebug("ExoticaValidation") << "In HLTExoticaValidator::beginJob()";
}

void HLTExoticaValidator::endRun(const edm::Run & iRun, const edm::EventSetup& iSetup)
{
}


void HLTExoticaValidator::endJob()
{
}

//define this as a plug-in
//DEFINE_FWK_MODULE(HLTExoticaValidator);
