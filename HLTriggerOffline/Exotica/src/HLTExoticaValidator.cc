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
	_collections(0),
	_dbe(0)
{
	_collections = new EVTColContainer;
}

HLTExoticaValidator::~HLTExoticaValidator()
{
	if( _collections != 0 )
	{
		delete _collections;
		_collections = 0;
	}
}


void HLTExoticaValidator::beginRun(const edm::Run & iRun, const edm::EventSetup & iSetup) 
{
	// Create a new subanalysis for each of the analysis names.
	// Notice that the constructor takes the full parameter set AND
	// the analysis separately.
	for(size_t i = 0; i < _analysisnames.size() ; ++i)
	{
		HLTExoticaSubAnalysis analyzer(_pset, _analysisnames.at(i));
		_analyzers.push_back(analyzer);
	}

	// Call the Plotter beginRun (which stores the triggers paths..:)
      	for(std::vector<HLTExoticaSubAnalysis>::iterator iter = _analyzers.begin(); 
			iter != _analyzers.end(); ++iter) 
	{
	    	iter->beginRun(iRun, iSetup);
	}
}
	

void HLTExoticaValidator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
      	static int eventNumber = 0;
      	eventNumber++;
      	LogTrace("ExoticaValidation") << "In HLTExoticaSubAnalysis::analyze,  " 
		<< "Event: " << eventNumber;
	
	// Initialize the event collections
	this->_collections->reset();

      	for(std::vector<HLTExoticaSubAnalysis>::iterator iter = _analyzers.begin(); 
			iter != _analyzers.end(); ++iter) 
	{
	     	iter->analyze(iEvent, iSetup, this->_collections);
      	}
}



void HLTExoticaValidator::beginJob()
{
}

void HLTExoticaValidator::endRun(const edm::Run & iRun, const edm::EventSetup& iSetup)
{
}


void HLTExoticaValidator::endJob()
{
}

//define this as a plug-in
//DEFINE_FWK_MODULE(HLTExoticaValidator);
