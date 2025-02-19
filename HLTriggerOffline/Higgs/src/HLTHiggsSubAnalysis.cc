
/** \file HLTHiggsSubAnalysis.cc
 *  $Date: 2012/05/02 11:46:23 $
 *  $Revision: 1.8 $
 */

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"

#include "HLTriggerOffline/Higgs/interface/HLTHiggsSubAnalysis.h"
#include "HLTriggerOffline/Higgs/src/EVTColContainer.cc"
#include "HLTriggerOffline/Higgs/src/MatchStruct.cc"

#include "FWCore/Common/interface/TriggerNames.h"

#include "TPRegexp.h"
#include "TString.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "HLTriggerOffline/Higgs/interface/HLTHiggsSubAnalysis.h"
#include "HLTriggerOffline/Higgs/src/MatchStruct.cc"

#include "TPRegexp.h"
#include "TString.h"

#include<set>
#include<algorithm>


HLTHiggsSubAnalysis::HLTHiggsSubAnalysis(const edm::ParameterSet & pset,
		const std::string & analysisname) :
	_pset(pset),
	_analysisname(analysisname),
	_minCandidates(0),
	_hltProcessName(pset.getParameter<std::string>("hltProcessName")),
	_genParticleLabel(pset.getParameter<std::string>("genParticleLabel")),
      	_parametersEta(pset.getParameter<std::vector<double> >("parametersEta")),
  	_parametersPhi(pset.getParameter<std::vector<double> >("parametersPhi")),
  	_parametersTurnOn(pset.getParameter<std::vector<double> >("parametersTurnOn")),
	_recMuonSelector(0),
	_recElecSelector(0),
	_recCaloMETSelector(0),
	_recPFTauSelector(0),
	_recPhotonSelector(0),
	_recTrackSelector(0),
	_dbe(0)
{
	// Specific parameters for this analysis
	edm::ParameterSet anpset = pset.getParameter<edm::ParameterSet>(analysisname);
	// Collections labels (but genparticles already initialized) 
	// initializing _recLabels data member)
	this->bookobjects( anpset );

	// Generic objects: Initialization of cuts
	for(std::map<unsigned int,std::string>::const_iterator it = _recLabels.begin();
			it != _recLabels.end(); ++it)
	{
		const std::string objStr = EVTColContainer::getTypeString(it->first);
		_genCut[it->first] = pset.getParameter<std::string>( std::string(objStr+"_genCut").c_str() );
		_recCut[it->first] = pset.getParameter<std::string>( std::string(objStr+"_recCut").c_str() );
		_cutMinPt[it->first] = pset.getParameter<double>( std::string(objStr+"_cutMinPt").c_str() );
		_cutMaxEta[it->first] = pset.getParameter<double>( std::string(objStr+"_cutMaxEta").c_str() );
	}

	//--- Updating parameters if has to be modified for this particular specific analysis
	for(std::map<unsigned int,std::string>::const_iterator it = _recLabels.begin();
			it != _recLabels.end(); ++it)
	{
		const std::string objStr = EVTColContainer::getTypeString(it->first);
		try
		{
			_genCut[it->first] = anpset.getUntrackedParameter<std::string>( std::string(objStr+"_genCut").c_str() ); 
		}
		catch(edm::Exception)
		{
		}
		try
		{
			_recCut[it->first] = anpset.getUntrackedParameter<std::string>( std::string(objStr+"_recCut").c_str() );
		}
		catch(edm::Exception)
		{
		}
		try
		{
			_cutMinPt[it->first] = anpset.getUntrackedParameter<double>( std::string(objStr+"_cutMinPt").c_str() );
		}
		catch(edm::Exception)
		{
		}
		try
		{
			_cutMaxEta[it->first] = anpset.getUntrackedParameter<double>( std::string(objStr+"_cutMaxEta").c_str() );
		}
		catch(edm::Exception)
		{
		}
	}
	
	_hltPathsToCheck = anpset.getParameter<std::vector<std::string> >("hltPathsToCheck");
	_minCandidates = anpset.getParameter<unsigned int>("minCandidates");

	_dbe = edm::Service<DQMStore>().operator->();
      	_dbe->setVerbose(0);
}

HLTHiggsSubAnalysis::~HLTHiggsSubAnalysis()
{
	for(std::map<unsigned int,StringCutObjectSelector<reco::GenParticle>* >::iterator it = _genSelectorMap.begin();
			it != _genSelectorMap.end(); ++it)
	{
		if( it->second != 0)
		{
			delete it->second;
			it->second =0;
		}
	}
	if( _recMuonSelector != 0)
	{
		delete _recMuonSelector;
		_recMuonSelector =0;
	}
	if( _recElecSelector != 0)
	{
		delete _recElecSelector;
		_recElecSelector =0;
	}
	if( _recPhotonSelector != 0)
	{
		delete _recPhotonSelector;
		_recPhotonSelector =0;
	}
	if( _recCaloMETSelector != 0)
	{
		delete _recCaloMETSelector;
		_recCaloMETSelector =0;
	}
	if( _recPFTauSelector != 0)
	{
		delete _recPFTauSelector;
		_recPFTauSelector =0;
	}
	if( _recTrackSelector != 0)
	{
		delete _recTrackSelector;
		_recTrackSelector =0;
	}
}


void HLTHiggsSubAnalysis::beginJob() 
{
}



void HLTHiggsSubAnalysis::beginRun(const edm::Run & iRun, const edm::EventSetup & iSetup)
{
	std::string baseDir = "HLT/Higgs/"+_analysisname+"/";
      	_dbe->setCurrentFolder(baseDir);

	// Initialize the confighlt
	bool changedConfig;
	if(!_hltConfig.init(iRun,iSetup,_hltProcessName,changedConfig))
	{
		edm::LogError("HiggsValidations") << "HLTHiggsSubAnalysis::beginRun: "
			<< "Initializtion of HLTConfigProvider failed!!";
	}


	// Parse the input paths to get them if there are in the table 
	// and associate them the last filter of the path (in order to extract the
	_hltPaths.clear();
	for(size_t i = 0; i < _hltPathsToCheck.size(); ++i)
	{
		bool found = false;
		TPRegexp pattern(_hltPathsToCheck[i]);
		for(size_t j = 0 ; j < _hltConfig.triggerNames().size(); ++j)
		{
			std::string thetriggername = _hltConfig.triggerNames()[j];
			if(TString(thetriggername).Contains(pattern))
			{
				_hltPaths.insert(thetriggername);
				found = true;
			}
		}
		if( ! found )
		{
			edm::LogWarning("HiggsValidations") << "HLTHiggsSubAnalysis::beginRun, In "
				<< _analysisname << " subfolder NOT found the path: '" 
				<< _hltPathsToCheck[i] << "*'" ;
		}
	}

	LogTrace("HiggsValidation") << "SubAnalysis: " << _analysisname 
		<< "\nHLT Trigger Paths found >>>"; 
      	// Initialize the plotters (analysers for each trigger path)
	_analyzers.clear();
  	for(std::set<std::string>::iterator iPath = _hltPaths.begin(); 
			iPath != _hltPaths.end(); ++iPath) 
	{
		// Avoiding the dependence of the version number for
		// the trigger paths
		std::string path = * iPath;
		std::string shortpath = path;
	    	if(path.rfind("_v") < path.length())
		{
			shortpath = path.substr(0, path.rfind("_v"));
		}
		_shortpath2long[shortpath] = path;

		// Objects needed by the HLT path
		const std::vector<unsigned int> objsNeedHLT = this->getObjectsType(shortpath);
		// Sanity check: the object needed by a trigger path should be
		// introduced by the user via config python (_recLabels datamember)
		std::vector<unsigned int> userInstantiate;
		for(std::map<unsigned int,std::string>::iterator it = _recLabels.begin() ; 
				it != _recLabels.end(); ++it)
		{
			userInstantiate.push_back(it->first);
		}
		for(std::vector<unsigned int>::const_iterator it = objsNeedHLT.begin(); it != objsNeedHLT.end(); 
				++it)
		{
			if( std::find(userInstantiate.begin(),userInstantiate.end(), *it) == 
					userInstantiate.end() )
			{
				edm::LogError("HiggsValidation") << "In HLTHiggsSubAnalysis::beginRun, " 
					<< "Incoherence found in the python configuration file!!\nThe SubAnalysis '" 
					<< _analysisname << "' has been asked to evaluate the trigger path '"
					<< shortpath << "' (found it in 'hltPathsToCheck') BUT this path"
					<< " needs a '" << EVTColContainer::getTypeString(*it) 
 					<< "' which has not been instantiate ('recVariableLabels'" 
					<< ")" ;
				exit(-1);
			}
		}
		LogTrace("HiggsValidation") << " --- " << shortpath;
		
		// the hlt path, the objects (elec,muons,photons,...)
		// needed to evaluate the path are the argumens of the plotter
		HLTHiggsPlotter analyzer(_pset, shortpath,objsNeedHLT, _dbe);
		_analyzers.push_back(analyzer);
    	}

      	// Call the beginRun (which books all the path dependent histograms)
      	for(std::vector<HLTHiggsPlotter>::iterator it = _analyzers.begin(); 
			it != _analyzers.end(); ++it) 
	{
	    	it->beginRun(iRun, iSetup);
	}

	// Book the gen/reco analysis-dependent histograms (denominators)
	for(std::map<unsigned int,std::string>::const_iterator it = _recLabels.begin();
			it != _recLabels.end(); ++it)
	{
		const std::string objStr = EVTColContainer::getTypeString(it->first);
		std::vector<std::string> sources(2);
		sources[0] = "gen";
		sources[1] = "rec";
	  
		for(size_t i = 0; i < sources.size(); i++) 
		{
			std::string source = sources[i];
			bookHist(source, objStr, "Eta");
			bookHist(source, objStr, "Phi");
			bookHist(source, objStr, "MaxPt1");
			bookHist(source, objStr, "MaxPt2");
		}
	}
}



void HLTHiggsSubAnalysis::analyze(const edm::Event & iEvent, const edm::EventSetup & iSetup, 
		EVTColContainer * cols)
{
	// Initialize the collection (the ones which hasn't been initialiazed yet)
 	this->initobjects(iEvent,cols);
	// utility map
	std::map<unsigned int,std::string> u2str;
	u2str[GEN]="gen";
	u2str[RECO]="rec";

	// Extract the match structure containing the gen/reco candidates (electron, muons,...)
	// common to all the SubAnalysis
	//---- Generation
 	// Make each good gen object into the base cand for a MatchStruct
	std::vector<MatchStruct> * matches = new std::vector<MatchStruct>;
//	bool alreadyMu = false;
	for(std::map<unsigned int,std::string>::iterator it = _recLabels.begin();
			it != _recLabels.end(); ++it)
	{
		// Avoiding the TkMu and Mu case
/*		if( alreadyMu )
		{
			continue;
		}*/
		// Initialize selectors when first event
		if(!_genSelectorMap[it->first]) 
		{
			_genSelectorMap[it->first] = new StringCutObjectSelector<reco::GenParticle>(_genCut[it->first]);
		}

		for(size_t i = 0; i < cols->genParticles->size(); ++i)
		{
			if(_genSelectorMap[it->first]->operator()(cols->genParticles->at(i)))
			{
				matches->push_back(MatchStruct(&cols->genParticles->at(i),it->first));
			}
		}
/*		if( it->first == EVTColContainer::MUON || it->first == EVTColContainer::TRACK )
		{
			alreadyMu = true;
		}*/
	}
	// Sort the MatchStructs by pT for later filling of turn-on curve
	std::sort(matches->begin(), matches->end(), matchesByDescendingPt());
	
	// Map to reference the source (gen/reco) with the recoCandidates
	std::map<unsigned int,std::vector<MatchStruct> > sourceMatchMap;  // To be a pointer to delete
	// --- Storing the generating candidates
	sourceMatchMap[GEN] = *matches;
	
	// Reuse the vector
	matches->clear();
	// --- same for RECO objects
	// Extraction of the objects candidates 
	for(std::map<unsigned int,std::string>::iterator it = _recLabels.begin();
			it != _recLabels.end(); ++it)
	{
		// Reco selectors (the function takes into account if it was instantiated 
		// before or not
		this->InitSelector(it->first);
		// -- Storing the matches 
		this->insertcandidates(it->first,cols,matches);
	}
	// Sort the MatchStructs by pT for later filling of turn-on curve
	std::sort(matches->begin(), matches->end(), matchesByDescendingPt());
	// --- Storing the reco candidates
	sourceMatchMap[RECO] = *matches;
	// --- All the objects are in place
	delete matches;  
	
	// -- Trigger Results
	const edm::TriggerNames trigNames = iEvent.triggerNames(*(cols->triggerResults));

	// Filling the histograms if pass the minimum amount of candidates needed by the analysis:
	// GEN + RECO CASE in the same loop
	for(std::map<unsigned int,std::vector<MatchStruct> >::iterator it = sourceMatchMap.begin(); 
			it != sourceMatchMap.end(); ++it)
	{
		// it->first: gen/reco   it->second: matches (std::vector<MatchStruc>)
		if( it->second.size() < _minCandidates )   // FIXME: A bug is potentially here: what about the mixed channels?
		{
			continue;
		}
		
		// Filling the gen/reco objects (eff-denominators): 
		// Just the first two different ones, if there are more
		std::map<unsigned int,int> * countobjects = new std::map<unsigned int,int>;
		// Initializing the count of the used object
		for(std::map<unsigned int,std::string>::iterator co = _recLabels.begin();
				co != _recLabels.end(); ++co)
		{
			countobjects->insert(std::pair<unsigned int,int>(co->first,0));
		}
		int counttotal = 0;
		const int totalobjectssize2 = 2*countobjects->size();
		for(size_t j = 0; j < it->second.size(); ++j)
		{
   			const unsigned int objType = it->second[j].objType;
			const std::string objTypeStr = EVTColContainer::getTypeString(objType);

			float pt  = (it->second)[j].pt; 
			float eta = (it->second)[j].eta;
			float phi = (it->second)[j].phi;
			
			this->fillHist(u2str[it->first],objTypeStr,"Eta",eta);
			this->fillHist(u2str[it->first],objTypeStr,"Phi",phi);
			if( (*countobjects)[objType] == 0 )
			{
				this->fillHist(u2str[it->first],objTypeStr,"MaxPt1",pt);
				// Filled the high pt ...
				++((*countobjects)[objType]);
				++counttotal;
			}
			else if( (*countobjects)[objType] == 1  )
			{
				this->fillHist(u2str[it->first],objTypeStr,"MaxPt2",pt);
				// Filled the second high pt ...
				++((*countobjects)[objType]);
				++counttotal;
			}
			else
			{
				// Already the minimum two objects has been filled, get out...
				if( counttotal == totalobjectssize2 )
				{
					break;
				}
			}				
		}
		delete countobjects;
	
		// Calling to the plotters analysis (where the evaluation of the different trigger paths are done)
		const std::string source = u2str[it->first];
		for(std::vector<HLTHiggsPlotter>::iterator an = _analyzers.begin();
				an != _analyzers.end(); ++an)
		{
			const std::string hltPath = _shortpath2long[an->gethltpath()];
			const bool ispassTrigger =  cols->triggerResults->accept(trigNames.triggerIndex(hltPath));
			an->analyze(ispassTrigger,source,it->second);
		}
	}
}

// Return the objects (muons,electrons,photons,...) needed by a hlt path. 
const std::vector<unsigned int> HLTHiggsSubAnalysis::getObjectsType(const std::string & hltPath) const
{
	static const unsigned int objSize = 5; //6;
	static const unsigned int objtriggernames[] = { EVTColContainer::MUON, 
		EVTColContainer::ELEC, 
		EVTColContainer::PHOTON,
//		EVTColContainer::TRACK,  // Note is tracker muon
	       	EVTColContainer::PFTAU,
		EVTColContainer::CALOMET
	};

	std::set<unsigned int> objsType;
	// The object to deal has to be entered via the config .py
	for(unsigned int i = 0; i < objSize; ++i)
	{
		std::string objTypeStr = EVTColContainer::getTypeString( objtriggernames[i] );
		// Check if it is needed this object for this trigger
		if( ! TString(hltPath).Contains(objTypeStr) )
		{
			continue;
		}

		objsType.insert(objtriggernames[i]);
	}

	return std::vector<unsigned int>(objsType.begin(),objsType.end());
}


// Booking the maps: recLabels and genParticle selectors
void HLTHiggsSubAnalysis::bookobjects( const edm::ParameterSet & anpset )
{
	if( anpset.exists("recMuonLabel") )
	{
		_recLabels[EVTColContainer::MUON] = anpset.getParameter<std::string>("recMuonLabel");
		_genSelectorMap[EVTColContainer::MUON] = 0 ;
	}
	if( anpset.exists("recElecLabel") )
	{
		_recLabels[EVTColContainer::ELEC] = anpset.getParameter<std::string>("recElecLabel");
		_genSelectorMap[EVTColContainer::ELEC] = 0 ;
	}
	if( anpset.exists("recPhotonLabel") )
	{
		_recLabels[EVTColContainer::PHOTON] = anpset.getParameter<std::string>("recPhotonLabel");
		_genSelectorMap[EVTColContainer::PHOTON] = 0 ;
	}
	if( anpset.exists("recCaloMETLabel") )
	{
		_recLabels[EVTColContainer::CALOMET] = anpset.getParameter<std::string>("recCaloMETLabel");
		_genSelectorMap[EVTColContainer::CALOMET] = 0 ;
	}
	if( anpset.exists("recPFTauLabel") )
	{
		_recLabels[EVTColContainer::PFTAU] = anpset.getParameter<std::string>("recPFTauLabel");
		_genSelectorMap[EVTColContainer::PFTAU] = 0 ;
	}
	/*if( anpset.exists("recTrackLabel") )
	{
		_recLabels[EVTColContainer::TRACK] = anpset.getParameter<std::string>("recTrackLabel");
		_genSelectorMap[EVTColContainer::TRACK] = 0 ;
	}*/

	if( _recLabels.size() < 1 )
	{
		edm::LogError("HiggsValidation") << "HLTHiggsSubAnalysis::bookobjects, " 
		<< "Not included any object (recMuonLabel, recElecLabel, ...)  "
	       	<< "in the analysis " << _analysisname;
		return;
	}
}

void HLTHiggsSubAnalysis::initobjects(const edm::Event & iEvent, EVTColContainer * col)
{
	/*if( col != 0 && col->isAllInit() )
	{
		// Already init, not needed to do nothing
		return;
	}*/
	if( ! col->isCommonInit() )
	{
		// TO BE DEPRECATED AS we don't need it anymore.
		// There is no match with HLT candidates... Use, then TriggerResults
		/*edm::Handle<trigger::TriggerEventWithRefs> rawTEH;
		iEvent.getByLabel("hltTriggerSummaryRAW",rawTEH);
		if(rawTEH.failedToGet())
		{
			edm::LogError("HiggsValidation") << "HLTHiggsSubAnalysis::initobjecst, "
				<< "No trigger summary found"; 
			return;
		}
		col->rawTriggerEvent = rawTEH.product();*/
		// END-- TO BE DEPRECATED
	
		// extract the trigger results (path info, pass,...)
		edm::Handle<edm::TriggerResults> trigResults;
		edm::InputTag trigResultsTag("TriggerResults","",_hltProcessName);
		iEvent.getByLabel(trigResultsTag,trigResults);
		if( trigResults.isValid() )
		{
			col->triggerResults = trigResults.product();
		}

		// GenParticle collection if is there
		edm::Handle<reco::GenParticleCollection> genPart;
		iEvent.getByLabel(_genParticleLabel,genPart);
		if( genPart.isValid() )
		{
			col->genParticles = genPart.product();
		}
	}
		
	for(std::map<unsigned int,std::string>::iterator it = _recLabels.begin(); 
			it != _recLabels.end(); ++it)
	{
		if( it->first == EVTColContainer::MUON )
		{
			edm::Handle<reco::MuonCollection> theHandle;
			iEvent.getByLabel(it->second, theHandle);
			col->set(theHandle.product());
		}
		else if( it->first == EVTColContainer::ELEC )
		{
			edm::Handle<reco::GsfElectronCollection> theHandle;
			iEvent.getByLabel(it->second, theHandle);
			col->set(theHandle.product());
		}
		else if( it->first == EVTColContainer::PHOTON )
		{
			edm::Handle<reco::PhotonCollection> theHandle;
			iEvent.getByLabel(it->second, theHandle);
			col->set(theHandle.product());
		}
		else if( it->first == EVTColContainer::CALOMET )
		{
			edm::Handle<reco::CaloMETCollection> theHandle;
			iEvent.getByLabel(it->second, theHandle);
			col->set(theHandle.product());
		}
		else if( it->first == EVTColContainer::PFTAU )
		{
			edm::Handle<reco::PFTauCollection> theHandle;
			iEvent.getByLabel(it->second, theHandle);
			col->set(theHandle.product());
		}
/*		else if( it->first == EVTColContainer::TRACK )
		{
			edm::Handle<reco::TrackCollection> theHandle;
			iEvent.getByLabel(it->second, theHandle);
			col->set(theHandle.product());
		}*/
		else
		{
			edm::LogError("HiggsValidation") << "HLTHiggsSubAnalysis::initobjects " 
				<< " NOT IMPLEMENTED (yet) ERROR: '" << it->second << "'";
			//return; ??
		}
	}
}

void HLTHiggsSubAnalysis::bookHist(const std::string & source, 
		const std::string & objType, const std::string & variable)
{
	std::string sourceUpper = source; 
      	sourceUpper[0] = std::toupper(sourceUpper[0]);
	std::string name = source + objType + variable ;
      	TH1F * h = 0;

      	if(variable.find("MaxPt") != std::string::npos) 
	{
		std::string desc = (variable == "MaxPt1") ? "Leading" : "Next-to-Leading";
		std::string title = "pT of " + desc + " " + sourceUpper + " " + objType;
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
		std::string title  = symbol + " of " + sourceUpper + " " + objType;
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

void HLTHiggsSubAnalysis::fillHist(const std::string & source, 
		const std::string & objType, const std::string & variable, const float & value )
{
	std::string sourceUpper = source; 
      	sourceUpper[0] = toupper(sourceUpper[0]);
	std::string name = source + objType + variable ;

	_elements[name]->Fill(value);
}



// Initialize the selectors
void HLTHiggsSubAnalysis::InitSelector(const unsigned int & objtype)
{	
	if( objtype == EVTColContainer::MUON && _recMuonSelector == 0 )
	{
		_recMuonSelector = new StringCutObjectSelector<reco::Muon>(_recCut[objtype]);
	}
	else if( objtype == EVTColContainer::ELEC && _recElecSelector == 0)
	{
		_recElecSelector = new StringCutObjectSelector<reco::GsfElectron>(_recCut[objtype]);
	}
	else if( objtype == EVTColContainer::PHOTON && _recPhotonSelector == 0)
	{
		_recPhotonSelector = new StringCutObjectSelector<reco::Photon>(_recCut[objtype]);
	}
	else if( objtype == EVTColContainer::CALOMET && _recCaloMETSelector == 0)
	{
		_recCaloMETSelector = new StringCutObjectSelector<reco::CaloMET>(_recCut[objtype]);
	}
	else if( objtype == EVTColContainer::PFTAU && _recPFTauSelector == 0 )
	{
		_recPFTauSelector = new StringCutObjectSelector<reco::PFTau>(_recCut[objtype]);
	}
	/*else if( objtype == EVTColContainer::TRACK && _recTrackSelector == 0)
	{
		_recTrackSelector = new StringCutObjectSelector<reco::Track>(_recCut[objtype]);
	}*/
/*	else
	{
FIXME: ERROR NO IMPLEMENTADO
	}*/
}

void HLTHiggsSubAnalysis::insertcandidates(const unsigned int & objType, const EVTColContainer * cols, 
		std::vector<MatchStruct> * matches)
{
	if( objType == EVTColContainer::MUON )
	{
		for(size_t i = 0; i < cols->muons->size(); i++)
		{
			if(_recMuonSelector->operator()(cols->muons->at(i)))
			{
				matches->push_back(MatchStruct(&cols->muons->at(i),objType));
			}
		}
	}
	else if( objType == EVTColContainer::ELEC )
	{
		for(size_t i = 0; i < cols->electrons->size(); i++)
		{
			if(_recElecSelector->operator()(cols->electrons->at(i)))
			{
				matches->push_back(MatchStruct(&cols->electrons->at(i),objType));
			}
		}
	}
	else if( objType == EVTColContainer::PHOTON )
	{
		for(size_t i = 0; i < cols->photons->size(); i++)
		{
			if(_recPhotonSelector->operator()(cols->photons->at(i)))
			{
				matches->push_back(MatchStruct(&cols->photons->at(i),objType));
			}
		}
	}
	else if( objType == EVTColContainer::CALOMET )
	{
		for(size_t i = 0; i < cols->caloMETs->size(); i++)
		{
			if(_recCaloMETSelector->operator()(cols->caloMETs->at(i)))
			{
				matches->push_back(MatchStruct(&cols->caloMETs->at(i),objType));
			}
		}
	}
	else if( objType == EVTColContainer::PFTAU )
	{
		for(size_t i = 0; i < cols->pfTaus->size(); i++)
		{
			if(_recPFTauSelector->operator()(cols->pfTaus->at(i)))
			{
				matches->push_back(MatchStruct(&cols->pfTaus->at(i),objType));
			}
		}
	}
	/*else if( objType == EVTColContainer::TRACK )
	{
		for(size_t i = 0; i < cols->tracks->size(); i++)
		{
			if(_recTrackSelector->operator()(cols->tracks->at(i)))
			{
				matches->push_back(MatchStruct(&cols->tracks->at(i),objType));
			}
		}
	}*/
	/*
	else FIXME: Control errores
	{
	}
	*/
}
