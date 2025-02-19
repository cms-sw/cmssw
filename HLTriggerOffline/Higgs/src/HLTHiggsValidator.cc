// -*- C++ -*-
//
// Package:     HLTHiggsValidator
// Class:       HLTHiggsValidator
// 

//
// Jordi Duarte Campderros (based on the Jason Slaunwhite 
// and Jeff Klukas coded from the HLTriggerOffline/Muon package
//
// $Id: HLTHiggsValidator.cc,v 1.7 2012/03/23 11:50:56 duarte Exp $
//
//

// system include files
//#include<memory>

//#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTriggerOffline/Higgs/interface/HLTHiggsValidator.h"
#include "HLTriggerOffline/Higgs/src/EVTColContainer.cc"

//////// Class Methods ///////////////////////////////////////////////////////
// Constructor
HLTHiggsValidator::HLTHiggsValidator(const edm::ParameterSet& pset) :
      	_pset(pset),
	_analysisnames(pset.getParameter<std::vector<std::string> >("analysis")),
	_collections(0),
	_dbe(0)
{
	_collections = new EVTColContainer;
}

HLTHiggsValidator::~HLTHiggsValidator()
{
	if( _collections != 0 )
	{
		delete _collections;
		_collections = 0;
	}
}


void HLTHiggsValidator::beginRun(const edm::Run & iRun, const edm::EventSetup & iSetup) 
{
	for(size_t i = 0; i < _analysisnames.size() ; ++i)
	{
		HLTHiggsSubAnalysis analyzer(_pset, _analysisnames.at(i));
		_analyzers.push_back(analyzer);
	}
	// Call the Plotter beginRun (which stores the triggers paths..:)
      	for(std::vector<HLTHiggsSubAnalysis>::iterator iter = _analyzers.begin(); 
			iter != _analyzers.end(); ++iter) 
	{
	    	iter->beginRun(iRun, iSetup);
	}
}
	

void HLTHiggsValidator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
      	static int eventNumber = 0;
      	eventNumber++;
      	LogTrace("HiggsValidation") << "In HLTHiggsSubAnalysis::analyze,  " 
		<< "Event: " << eventNumber;
	
	// Initialize the event collections
	this->_collections->reset();

      	for(std::vector<HLTHiggsSubAnalysis>::iterator iter = _analyzers.begin(); 
			iter != _analyzers.end(); ++iter) 
	{
	     	iter->analyze(iEvent, iSetup, this->_collections);
      	}
}



void HLTHiggsValidator::beginJob()
{
}

void HLTHiggsValidator::endRun(const edm::Run & iRun, const edm::EventSetup& iSetup)
{
      	// vector<HLTMuonPlotter>::iterator iter;
      	// for(std::vector<HLTHiggsPlotter>::iterator iter = _analyzers.begin(); 
	//                 iter != analyzers_.end(); ++iter) 
	// {
      	//         iter->endRun(iRun, iSetup);
      	// }
}


void HLTHiggsValidator::endJob()
{
}



//define this as a plug-in
//DEFINE_FWK_MODULE(HLTHiggsValidator);
