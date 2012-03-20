
/** \file HLTHiggsPlotter.cc
 *  $Date: 2012/03/16 01:55:33 $
 *  $Revision: 1.2 $
 */


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "HLTriggerOffline/Higgs/interface/HLTHiggsPlotter.h"
#include "HLTriggerOffline/Higgs/interface/HLTHiggsSubAnalysis.h"
#include "HLTriggerOffline/Higgs/src/EVTColContainer.cc"

#include "TPRegexp.h"


#include<set>
#include<cctype>

HLTHiggsPlotter::HLTHiggsPlotter(const edm::ParameterSet & pset,
		const std::string & hltPath,
		const std::string & lastfilter,
		const std::vector<unsigned int> & objectsType,
	       	DQMStore * dbe) :
	_hltPath(hltPath),
	_lastFilter(lastfilter),
	_hltProcessName(pset.getParameter<std::string>("hltProcessName")),
	_objectsType(std::set<unsigned int>(objectsType.begin(),objectsType.end())),
	_nObjects(objectsType.size()),
      	_parametersEta(pset.getParameter<std::vector<double> >("parametersEta")),
  	_parametersPhi(pset.getParameter<std::vector<double> >("parametersPhi")),
  	_parametersTurnOn(pset.getParameter<std::vector<double> >("parametersTurnOn")),
	_dbe(dbe)
{
	for(std::set<unsigned int>::iterator it = _objectsType.begin();
			it != _objectsType.end(); ++it)
	{
		// Some parameters extracted from the .py
		std::string objStr = this->getTypeString( *it );
		_cutMinPt[*it] = pset.getParameter<double>( std::string(objStr+"_cutMinPt").c_str() );
		_cutMaxEta[*it] = pset.getParameter<double>( std::string(objStr+"_cutMaxEta").c_str() );
		_cutMotherId[*it] = pset.getParameter<unsigned int>( std::string(objStr+"_cutMotherId").c_str() );
		_cutsDr[*it] = pset.getParameter<std::vector<double> >( std::string(objStr+"_cutDr").c_str() );
	}
}

HLTHiggsPlotter::~HLTHiggsPlotter()
{
}


void HLTHiggsPlotter::beginJob() 
{
}



void HLTHiggsPlotter::beginRun(const edm::Run & iRun, const edm::EventSetup & iSetup)
{
	static int runNumber = 0;
      	runNumber++;

	for(std::set<unsigned int>::iterator it = _objectsType.begin(); 
			it != _objectsType.end(); ++it)
	{
		std::vector<std::string> sources(2);
		sources[0] = "gen";
		sources[1] = "rec";

		const std::string objTypeStr = this->getTypeString(*it);
	  
		for(size_t i = 0; i < sources.size(); i++) 
		{
			std::string source = sources[i];
			bookHist(source, objTypeStr, "Eta");
			bookHist(source, objTypeStr, "Phi");
			bookHist(source, objTypeStr, "MaxPt1");
			bookHist(source, objTypeStr, "MaxPt2");
		}
	}
}

void HLTHiggsPlotter::analyze(const bool & isPassTrigger, const std::string & source,
	       	const std::vector<MatchStruct> & matches)
{
	if( ! isPassTrigger )
	{
		return;
	}

	// Fill the histos if pass the trigger (just the two with higher pt)
	for(size_t j = 0; j < matches.size(); ++j)
	{
		std::string objTypeStr = this->getTypeString(matches[j].objType);

		float pt  = matches[j].candBase->pt();
		float eta = matches[j].candBase->eta();
		float phi = matches[j].candBase->phi();
		this->fillHist(isPassTrigger,source,objTypeStr,"Eta",eta);
		this->fillHist(isPassTrigger,source,objTypeStr,"Phi",phi);
		if( j == 0 )
		{
			this->fillHist(isPassTrigger,source,objTypeStr,"MaxPt1",pt);
		}
		else if( j == 1 )
		{
			this->fillHist(isPassTrigger,source,objTypeStr,"MaxPt2",pt);
			break;
		}
	}
}


void HLTHiggsPlotter::bookHist(const std::string & source, 
		const std::string & objType, const std::string & variable)
{
	std::string sourceUpper = source; 
      	sourceUpper[0] = std::toupper(sourceUpper[0]);
	std::string name = source + objType + variable + "_" + _hltPath;
      	TH1F * h = 0;

      	if(variable.find("MaxPt") != std::string::npos) 
	{
		std::string desc = (variable == "MaxPt1") ? "Leading" : "Next-to-Leading";
		std::string title = "pT of " + desc + " " + sourceUpper + " " + objType + " "
                   "where event pass the "+ _hltPath;
	    	const size_t nBins = _parametersTurnOn.size() - 1;
	    	float * edges = new float[nBins + 1];
	    	for(size_t i = 0; i < nBins + 1; i++)
		{
			edges[i] = _parametersTurnOn[i];
		}
	    	h = new TH1F(name.c_str(), title.c_str(), nBins, edges);
		delete edges;
      	}
      	else 
	{
		std::string symbol = (variable == "Eta") ? "#eta" : "#phi";
		std::string title  = symbol + " of " + sourceUpper + " " + objType + " "+
     			"where event pass the "+ _hltPath;
		std::vector<double> params = (variable == "Eta") ? _parametersEta : _parametersPhi;

	    	int    nBins = (int)params[0];
	    	double min   = params[1];
	    	double max   = params[2];
	    	h = new TH1F(name.c_str(), title.c_str(), nBins, min, max);
      	}
      	h->Sumw2();
      	_elements[name] = _dbe->book1D(name, h);
      	delete h;
}

void HLTHiggsPlotter::fillHist(const bool & passTrigger, const std::string & source, 
		const std::string & objType, const std::string & variable, const float & value )
{
	std::string sourceUpper = source; 
      	sourceUpper[0] = toupper(sourceUpper[0]);
	std::string name = source + objType + variable + "_" + _hltPath;

	_elements[name]->Fill(value);
}


//! 
const std::string HLTHiggsPlotter::getTypeString(const unsigned int & objtype) const
{
	std::string objTypestr("Mu");

	if( objtype == HLTHiggsSubAnalysis::ELEC )
	{
		objTypestr = "Ele";
	}
	else if( objtype == HLTHiggsSubAnalysis::PHOTON )
	{
		objTypestr = "Photon";
	}
	else if( objtype == HLTHiggsSubAnalysis::CALOMET )
	{
		objTypestr = "CaloMET";
	}
	else if( objtype == HLTHiggsSubAnalysis::PFTAU )
	{
		objTypestr = "PFTau";
	}
	/*else
	{ ERROR FIXME
	}*/

	return objTypestr;
}
