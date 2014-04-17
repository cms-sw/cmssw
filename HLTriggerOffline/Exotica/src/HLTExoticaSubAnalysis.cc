
/** \file HLTExoticaSubAnalysis.cc
 */

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "DataFormats/Common/interface/Handle.h"

#include "HLTriggerOffline/Exotica/interface/HLTExoticaSubAnalysis.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "CommonTools/Utils/interface/PtComparator.h"
#include "HLTriggerOffline/Exotica/src/EVTColContainer.cc"

#include "TPRegexp.h"
#include "TString.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include <set>
#include <algorithm>

/// Constructor
HLTExoticaSubAnalysis::HLTExoticaSubAnalysis(const edm::ParameterSet & pset,
                                             const std::string & analysisname,
					     edm::ConsumesCollector && consCollector) :
    _pset(pset),
    _analysisname(analysisname),
    _minCandidates(0),
    _hltProcessName(pset.getParameter<std::string>("hltProcessName")),
    _genParticleLabel(pset.getParameter<std::string>("genParticleLabel")),
    _trigResultsLabel("TriggerResults", "", _hltProcessName),
    _parametersEta(pset.getParameter<std::vector<double> >("parametersEta")),
    _parametersPhi(pset.getParameter<std::vector<double> >("parametersPhi")),
    _parametersTurnOn(pset.getParameter<std::vector<double> >("parametersTurnOn")),
    _recMuonSelector(0),
    _recElecSelector(0),
    _recPFMETSelector(0),
    _recPFTauSelector(0),
    _recPhotonSelector(0),
    _recJetSelector(0)
{

    LogDebug("ExoticaValidation") << "In HLTExoticaSubAnalysis::constructor()";

    // Specific parameters for this analysis
    edm::ParameterSet anpset = pset.getParameter<edm::ParameterSet>(analysisname);

    // If this analysis has a particular set of binnings, use it.
    // (Taken from the analysis-specific parameter set, of course)
    // The "true" in the beginning of _pset.insert() means
    // "overwrite the parameter if need be".
    if (anpset.exists("parametersTurnOn")) {
        _parametersTurnOn = anpset.getParameter<std::vector<double> >("parametersTurnOn");
        _pset.insert(true, "parametersTurnOn", anpset.retrieve("parametersTurnOn"));
    }
    if (anpset.exists("parametersEta")) {
        _parametersEta = anpset.getParameter<std::vector<double> >("parametersEta");
        _pset.insert(true, "parametersEta", anpset.retrieve("parametersEta"));
    }
    if (anpset.exists("parametersPhi")) {
        _parametersPhi = anpset.getParameter<std::vector<double> >("parametersPhi");
        _pset.insert(true, "parametersPhi", anpset.retrieve("parametersPhi"));
    }

    // Get names of objects that we may want to get from the event.
    // Notice that genParticles are dealt with separately.
    this->getNamesOfObjects(anpset);
    
    // Since now we have the names, we should register the consumption
    // of objects.
    this->registerConsumes(consCollector);

    // Generic objects: Initialization of basic phase space cuts.
    for (std::map<unsigned int, edm::InputTag>::const_iterator it = _recLabels.begin();
         it != _recLabels.end(); ++it) {
	const std::string objStr = EVTColContainer::getTypeString(it->first);
        _genCut[it->first] = pset.getParameter<std::string>(std::string(objStr + "_genCut").c_str());
        _recCut[it->first] = pset.getParameter<std::string>(std::string(objStr + "_recCut").c_str());
    }

    //--- Updating parameters if has to be modified for this particular specific analysis
    for (std::map<unsigned int, edm::InputTag>::const_iterator it = _recLabels.begin();
         it != _recLabels.end(); ++it) {
	const std::string objStr = EVTColContainer::getTypeString(it->first);

        try {
            _genCut[it->first] = anpset.getUntrackedParameter<std::string>(std::string(objStr + "_genCut").c_str());
        } catch (edm::Exception) {}

        try {
            _recCut[it->first] = anpset.getUntrackedParameter<std::string>(std::string(objStr + "_recCut").c_str());
        } catch (edm::Exception) {}

    }

    /// Get the vector of paths to check, for this particular analysis.
    _hltPathsToCheck = anpset.getParameter<std::vector<std::string> >("hltPathsToCheck");
    /// Get the minimum candidates, for this particular analysis.
    _minCandidates = anpset.getParameter<unsigned int>("minCandidates");

} /// End Constructor

HLTExoticaSubAnalysis::~HLTExoticaSubAnalysis()
{
    for (std::map<unsigned int, StringCutObjectSelector<reco::GenParticle>* >::iterator it = _genSelectorMap.begin();
         it != _genSelectorMap.end(); ++it) {
        delete it->second;
        it->second = 0;
    }
    delete _recMuonSelector;
    _recMuonSelector = 0;
    delete _recElecSelector;
    _recElecSelector = 0;
    delete _recPhotonSelector;
    _recPhotonSelector = 0;
    delete _recPFMETSelector;
    _recPFMETSelector = 0;
    delete _recPFTauSelector;
    _recPFTauSelector = 0;
    delete _recJetSelector;
    _recJetSelector = 0;
}


void HLTExoticaSubAnalysis::beginJob()
{
}


// 2014-02-03 -- Thiago
// Due to the fact that the DQM has to be thread safe now, we have to do things differently:
// 1) Implement the bookHistograms() method in the container class
// 2) Make the iBooker from above be known to this class
// 3) Separate all booking histograms routines in this and any auxiliary classe to be called
// from bookHistograms() in the container class
void HLTExoticaSubAnalysis::subAnalysisBookHistos(DQMStore::IBooker &iBooker,
                                                  const edm::Run & iRun,
                                                  const edm::EventSetup & iSetup)
{
    
    LogDebug("ExoticaValidation") << "In HLTExoticaSubAnalysis::subAnalysisBookHistos()";

    // Create the folder structure inside HLT/Exotica
    std::string baseDir = "HLT/Exotica/" + _analysisname + "/";
    iBooker.setCurrentFolder(baseDir);

    // Book the gen/reco analysis-dependent histograms (denominators)
    for (std::map<unsigned int, edm::InputTag>::const_iterator it = _recLabels.begin();
         it != _recLabels.end(); ++it) {
	const std::string objStr = EVTColContainer::getTypeString(it->first);
        std::vector<std::string> sources(2);
        sources[0] = "gen";
        sources[1] = "rec";
	
        for (size_t i = 0; i < sources.size(); i++) {
            std::string source = sources[i];
            bookHist(iBooker, source, objStr, "Eta");
            bookHist(iBooker, source, objStr, "Phi");
            bookHist(iBooker, source, objStr, "MaxPt1");
            bookHist(iBooker, source, objStr, "MaxPt2");
            bookHist(iBooker, source, objStr, "SumEt");
        }
    } // closes loop in _recLabels

    // Call the plotterBookHistos() (which books all the path dependent histograms)
    LogDebug("ExoticaValidation") << "                        number of plotters = " << _plotters.size();
    for (std::vector<HLTExoticaPlotter>::iterator it = _plotters.begin();
         it != _plotters.end(); ++it) {
	it->plotterBookHistos(iBooker, iRun, iSetup);
    }
}

void HLTExoticaSubAnalysis::beginRun(const edm::Run & iRun, const edm::EventSetup & iSetup)
{

    LogDebug("ExoticaValidation") << "In HLTExoticaSubAnalysis::beginRun()";
    
    /// Construct the plotters right here.
    /// For that we need to create the _hltPaths vector.

    // Initialize the HLT config.
    bool changedConfig(true);
    if (!_hltConfig.init(iRun, iSetup, _hltProcessName, changedConfig)) {
        edm::LogError("ExoticaValidation") << "HLTExoticaSubAnalysis::constructor(): "
                                           << "Initialization of HLTConfigProvider failed!";
    }

    // Parse the input paths to get them if they are in the table and associate
    // them to the last filter of the path (in order to extract the objects).
    _hltPaths.clear();
    for (size_t i = 0; i < _hltPathsToCheck.size(); ++i) {
        bool found = false;
        TPRegexp pattern(_hltPathsToCheck[i]);
	
	// Loop over triggerNames from _hltConfig
	for (size_t j = 0 ; j < _hltConfig.triggerNames().size(); ++j) {
            std::string thetriggername = _hltConfig.triggerNames()[j];
            if (TString(thetriggername).Contains(pattern)) {
                _hltPaths.insert(thetriggername);
                found = true;
            }
        }
	
	// Oh dear, the path we wanted seems to not be available
        if (! found) {
            edm::LogWarning("ExoticaValidation") << "HLTExoticaSubAnalysis::constructor(): In "
                                                 << _analysisname << " subfolder NOT found the path: '"
                                                 << _hltPathsToCheck[i] << "*'" ;
        }
    } // Close loop over paths to check.
    
    // At this point, _hltpaths contains the names of the paths to check
    // that were found. Let's log it at trace level.
    LogTrace("ExoticaValidation") << "SubAnalysis: " << _analysisname 
				  << "\nHLT Trigger Paths found >>>";
    for (std::set<std::string>::const_iterator iter = _hltPaths.begin();
	 iter != _hltPaths.end(); ++iter) {
        LogTrace("ExoticaValidation") << (*iter) << "\n";
    }

    // Initialize the plotters (analysers for each trigger path)
    _plotters.clear();
    for (std::set<std::string>::iterator iPath = _hltPaths.begin();
         iPath != _hltPaths.end(); ++iPath) {
        // Avoiding the dependence of the version number for the trigger paths
        std::string path = * iPath;
        std::string shortpath = path;
        if (path.rfind("_v") < path.length()) {
            shortpath = path.substr(0, path.rfind("_v"));
        }
        _shortpath2long[shortpath] = path;

        // Objects needed by the HLT path
	// Thiago: instead of trying to decode the objects from the path,
	// put the burden on the user to tell us which objects are needed.
        //const std::vector<unsigned int> objsNeedHLT = this->getObjectsType(shortpath);
	std::vector<unsigned int> objsNeedHLT;
        for (std::map<unsigned int, edm::InputTag>::iterator it = _recLabels.begin() ;
             it != _recLabels.end(); ++it) {
            objsNeedHLT.push_back(it->first);
        }

        /*std::vector<unsigned int> userInstantiate;
        // Sanity check: the object needed by a trigger path should be
        // introduced by the user via config python (_recLabels datamember)
        for (std::map<unsigned int, edm::InputTag>::iterator it = _recLabels.begin() ;
             it != _recLabels.end(); ++it) {
            userInstantiate.push_back(it->first);
        }
        for (std::vector<unsigned int>::const_iterator it = objsNeedHLT.begin(); it != objsNeedHLT.end();
             ++it) {
            if (std::find(userInstantiate.begin(), userInstantiate.end(), *it) ==
                userInstantiate.end()) {
                edm::LogError("ExoticaValidation") << "In HLTExoticaSubAnalysis::beginRun, "
                                                   << "Incoherence found in the python configuration file!!\nThe SubAnalysis '"
                                                   << _analysisname << "' has been asked to evaluate the trigger path '"
                                                   << shortpath << "' (found it in 'hltPathsToCheck') BUT this path"
                                                   << " needs a '" << EVTColContainer::getTypeString(*it)
                                                   << "' which has not been instantiated ('recVariableLabels'"
                                                   << ")" ;
                exit(-1); // This should probably throw an exception...
            }
        }
	*/
        LogTrace("ExoticaValidation") << " --- " << shortpath;

        // The hlt path, the objects (electrons, muons, photons, ...)
        // needed to evaluate the path are the argumens of the plotter
        HLTExoticaPlotter analyzer(_pset, shortpath, objsNeedHLT);
        _plotters.push_back(analyzer);
    }// Okay, at this point we have prepared all the plotters.

}


void HLTExoticaSubAnalysis::analyze(const edm::Event & iEvent, const edm::EventSetup & iSetup, EVTColContainer * cols)
{
    LogDebug("ExoticaValidation") << "In HLTExoticaSubAnalysis::analyze()";

    // Loop over _recLabels to make sure everything is alright.
    /*
    std::cout << "Now printing the _recLabels" << std::endl;
    for (std::map<unsigned int, edm::InputTag>::iterator it = _recLabels.begin();
         it != _recLabels.end(); ++it) {
	std::cout <<  "Number: " << it->first << "\t" << "Label: " << it->second.label() << std::endl;  
    }
    */
    
    // Initialize the collection (the ones which have not been initialiazed yet)
    //std::cout << "Setting handles to objects..." << std::endl;
    this->getHandlesToObjects(iEvent, cols);

    // Utility map, mapping kinds of objects (GEN, RECO) to strings ("gen","rec")
    //std::map<Level, std::string> u2str;
    //u2str[Level::GEN] = "gen";
    //u2str[Level::RECO] = "rec";

    // Extract the match structure containing the gen/reco candidates (electron, muons,...). This part is common to all the SubAnalyses

    //std::map<unsigned int, std::vector<MatchStruct> > sourceMatchMap;
    //std::vector<MatchStruct> matchesGen; matchesGen.clear();
    std::vector<reco::LeafCandidate> matchesGen; matchesGen.clear();
    //std::vector<MatchStruct> matchesReco; matchesReco.clear();
    std::vector<reco::LeafCandidate> matchesReco; matchesReco.clear();
    //sourceMatchMap[98] = matchesGen;
    //sourceMatchMap[99] = matchesReco;

    //std::cout << "In the beginning: matchesGen.size() = " << matchesGen.size() << std::endl;
    //std::cout << "In the beginning: matchesReco.size() = " << matchesReco.size() << std::endl;
    
    // --- deal with GEN objects first.
    // Make each good GEN object into the base cand for a MatchStruct
    // Our definition of "good" is "passes the selector" defined in the config.py
    // Save all the MatchStructs in the "matchesGen" vector.

    for (std::map<unsigned int, edm::InputTag>::iterator it = _recLabels.begin();
         it != _recLabels.end(); ++it) {
        // Here we are filling the vector of StringCutObjectSelector<reco::GenParticle>
        // with objects constructed from the strings saved in _genCut.
        // Initialize selectors when first event

	//std::cout << "Loop over the kinds of objects: objects of kind " << it->first << std::endl;
	
	if (!_genSelectorMap[it->first]) {
            _genSelectorMap[it->first] = new StringCutObjectSelector<reco::GenParticle>(_genCut[it->first]);
        }

        // Now loop over the genParticles, and apply the operator() over each of them.
        // Fancy syntax: for objects X and Y, X.operator()(Y) is the same as X(Y).
        for (size_t i = 0; i < cols->genParticles->size(); ++i) {
	    //std::cout << "Now matchesGen.size() is " << matchesGen.size() << std::endl;
            if (_genSelectorMap[it->first]->operator()(cols->genParticles->at(i))) {
                const reco::Candidate* cand = &(cols->genParticles->at(i));
		//std::cout << "Found good cand: cand->pt() = " << cand->pt() << std::endl;
		//matchesGen.push_back(MatchStruct(cand, it->first));
		/// We are going to make a fake LeafCandidate, with our particleType as the pdgId.
		/// This is an alternative to the older implementation with MatchStruct.
		reco::LeafCandidate v(0,cand->p4(),cand->vertex(),it->first,0,true);
		matchesGen.push_back(v);
            }
        }
    }

    // Sort the matches by pT for later filling of turn-on curve
    //std::cout << "Before sorting: matchesGen.size() = " << matchesGen.size() << std::endl;
    GreaterByPt<reco::LeafCandidate> comparator;
    std::sort(matchesGen.begin(), 
	      matchesGen.end(), 
	      comparator);
    //std::cout << "After sorting: matchesGen.size() = " << matchesGen.size() << std::endl;

    //std::cout << "Before adding RECO: sourceMatchMap[98].size() = " << sourceMatchMap[98].size() << std::endl;
    //std::cout << "Before adding RECO: matchesGen.size() = " << matchesGen.size() << std::endl;

    // --- same for RECO objects
    // Extraction of the objects candidates
    for (std::map<unsigned int, edm::InputTag>::iterator it = _recLabels.begin();
         it != _recLabels.end(); ++it) {
	//std::cout << "Filling RECO \"matchesReco\" vector for particle kind it->first = "
	//	  << it->first << ", which means " << it->second.label() << std::endl;
        // Reco selectors (the function takes into account if it was instantiated
        // before or not) ### Thiago ---> Then why don't we put it in the beginRun???
        this->initSelector(it->first);
        // -- Storing the matchesReco
        this->insertCandidates(it->first, cols, &matchesReco);
    }

    //std::cout << "After adding RECO: matchesGen.size() = " << matchesGen.size() << std::endl;
    //std::cout << "After adding RECO: matchesReco.size() = " << matchesReco.size() << std::endl;

    std::sort(matchesReco.begin(), 
	      matchesReco.end(), 
	      comparator);

    // --- All the objects are in place
    //std::cout << "DEBUG(0)" << std::endl;

    // -- Trigger Results
    const edm::TriggerNames trigNames = iEvent.triggerNames(*(cols->triggerResults));

    /// Filling the histograms if pass the minimum amount of candidates needed by the analysis:
    /// First do the GEN case

    //for (std::map<unsigned int, std::vector<MatchStruct> >::iterator it = sourceMatchMap.begin(); it != sourceMatchMap.end(); ++it) {
    // it->first: gen/reco   it->second: HLT matches (std::vector<MatchStruct>)

    //if (it->second.size() < _minCandidates) {  // FIXME: A bug is potentially here: what about the mixed channels?
    //continue;
    //}

    //////////////// 
    /// GEN CASE ///
    //////////////// 
    {
	if(matchesGen.size() < _minCandidates) return; // FIXME: A bug is potentially here: what about the mixed channels?

	// Okay, there are enough candidates. Move on!
    
	// Filling the gen/reco objects (eff-denominators):
	// Just the first two different ones, if there are more
	// The countobjects maps uints (object types, really) --> integers.
	// Example:
	// | uint | int |
	// |  0   |  1  | --> 1 muon used
	// |  1   |  2  | --> 2 electrons used  
	// Initializing the count of the used objects.
	std::map<unsigned int, int> * countobjects = new std::map<unsigned int, int>;
	for (std::map<unsigned int, edm::InputTag>::iterator co = _recLabels.begin();
	     co != _recLabels.end(); ++co) {
	    countobjects->insert(std::pair<unsigned int, int>(co->first, 0));
	}
    
	int counttotal = 0;
	int totalobjectssize2 = 2 * countobjects->size();
    
	for (size_t j = 0; j != matchesGen.size(); ++j) {
	    const unsigned int objType = matchesGen[j].pdgId();
	    //std::cout << "(4) Gonna call with " << objType << std::endl;
	    const std::string objTypeStr = EVTColContainer::getTypeString(objType);
	
	    float pt  = matchesGen[j].pt();
	    float eta = matchesGen[j].eta();
	    float phi = matchesGen[j].phi();
	    float sumEt = 0;//matchesGen[j].sumEt;
	
	    this->fillHist("gen", objTypeStr, "Eta", eta);
	    this->fillHist("gen", objTypeStr, "Phi", phi);
	    this->fillHist("gen", objTypeStr, "SumEt", sumEt);

	    if ((*countobjects)[objType] == 0) {
		this->fillHist("gen", objTypeStr, "MaxPt1", pt);
		// Filled the high pt ...
		++((*countobjects)[objType]);
		++counttotal;
	    } else if ((*countobjects)[objType] == 1) {
		this->fillHist("gen", objTypeStr, "MaxPt2", pt);
		// Filled the second high pt ...
		++((*countobjects)[objType]);
		++counttotal;
	    } else {
		// Already the minimum two objects has been filled, get out...
		if (counttotal == totalobjectssize2) {
		    break;
		}
	    }
	} // Closes loop in gen

	LogDebug("ExoticaValidation") << "                        deleting countobjects";
	//delete countobjects;

	// Calling to the plotters analysis (where the evaluation of the different trigger paths are done)
	//const std::string source = "gen";
	for (std::vector<HLTExoticaPlotter>::iterator an = _plotters.begin(); an != _plotters.end(); ++an) {
	    const std::string hltPath = _shortpath2long[an->gethltpath()];
	    const bool ispassTrigger =  cols->triggerResults->accept(trigNames.triggerIndex(hltPath));
	    LogDebug("ExoticaValidation") << "                        preparing to call the plotters analysis";
	    an->analyze(ispassTrigger, "gen", matchesGen);
	    LogDebug("ExoticaValidation") << "                        called the plotter";
	}
    } /// Close GEN case


    ///////////////// 
    /// RECO CASE ///
    ///////////////// 
    {
	if(matchesReco.size() < _minCandidates) return; // FIXME: A bug is potentially here: what about the mixed channels?

	// Okay, there are enough candidates. Move on!
    
	// Filling the gen/reco objects (eff-denominators):
	// Just the first two different ones, if there are more
	// The countobjects maps uints (object types, really) --> integers.
	// Example:
	// | uint | int |
	// |  0   |  1  | --> 1 muon used
	// |  1   |  2  | --> 2 electrons used  
	// Initializing the count of the used objects.
	std::map<unsigned int, int> * countobjects = new std::map<unsigned int, int>;
	for (std::map<unsigned int, edm::InputTag>::iterator co = _recLabels.begin();
	     co != _recLabels.end(); ++co) {
	    countobjects->insert(std::pair<unsigned int, int>(co->first, 0));
	}
    
	int counttotal = 0;
	int totalobjectssize2 = 2 * countobjects->size();
    
	/// Debugging.
	//std::cout << "Our RECO vector has matchesReco.size() = " << matchesReco.size() << std::endl;
	
	for (size_t j = 0; j != matchesReco.size(); ++j) {
	    const unsigned int objType = matchesReco[j].pdgId();
	    //std::cout << "(4) Gonna call with " << objType << std::endl;
	    const std::string objTypeStr = EVTColContainer::getTypeString(objType);
	    
	    float pt  = matchesReco[j].pt();
	    float eta = matchesReco[j].eta();
	    float phi = matchesReco[j].phi();
	    float sumEt = 0;//matchesReco[j].sumEt;
	
	    this->fillHist("rec", objTypeStr, "Eta", eta);
	    this->fillHist("rec", objTypeStr, "Phi", phi);
	    this->fillHist("rec", objTypeStr, "SumEt", sumEt);

	    if ((*countobjects)[objType] == 0) {
		this->fillHist("rec", objTypeStr, "MaxPt1", pt);
		// Filled the high pt ...
		++((*countobjects)[objType]);
		++counttotal;
	    } else if ((*countobjects)[objType] == 1) {
		this->fillHist("rec", objTypeStr, "MaxPt2", pt);
		// Filled the second high pt ...
		++((*countobjects)[objType]);
		++counttotal;
	    } else {
		// Already the minimum two objects has been filled, get out...
		if (counttotal == totalobjectssize2) {
		    break;
		}
	    }
	} // Closes loop in reco

	LogDebug("ExoticaValidation") << "                        deleting countobjects";
	//delete countobjects;
	
	// Calling to the plotters analysis (where the evaluation of the different trigger paths are done)
	//const std::string source = "reco";
	for (std::vector<HLTExoticaPlotter>::iterator an = _plotters.begin(); an != _plotters.end(); ++an) {
	    const std::string hltPath = _shortpath2long[an->gethltpath()];
	    const bool ispassTrigger =  cols->triggerResults->accept(trigNames.triggerIndex(hltPath));
	    LogDebug("ExoticaValidation") << "                        preparing to call the plotters analysis";
	    an->analyze(ispassTrigger, "rec", matchesReco);
	    LogDebug("ExoticaValidation") << "                        called the plotter";
	}
    } /// Close RECO case

} /// closes analyze method


// Return the objects (muons,electrons,photons,...) needed by a hlt path.
const std::vector<unsigned int> HLTExoticaSubAnalysis::getObjectsType(const std::string & hltPath) const
{
    LogDebug("ExoticaValidation") << "In HLTExoticaSubAnalysis::getObjectsType()";

    static const unsigned int objSize = 6;
    static const unsigned int objtriggernames[] = {
        EVTColContainer::MUON,
        EVTColContainer::ELEC,
        EVTColContainer::PHOTON,
        EVTColContainer::PFMET,
        EVTColContainer::PFTAU,
        EVTColContainer::JET
    };

    std::set<unsigned int> objsType;
    // The object to deal has to be entered via the config .py
    for (unsigned int i = 0; i < objSize; ++i) {
	//std::cout << "(5) Gonna call with " << objtriggernames[i] << std::endl;
        std::string objTypeStr = EVTColContainer::getTypeString(objtriggernames[i]);
        // Check if it is needed this object for this trigger
        if (! TString(hltPath).Contains(objTypeStr)) {
            continue;
        }

        objsType.insert(objtriggernames[i]);
    }

    return std::vector<unsigned int>(objsType.begin(), objsType.end());
}

// Booking the maps: recLabels and genParticle selectors
void HLTExoticaSubAnalysis::getNamesOfObjects(const edm::ParameterSet & anpset)
{
    LogDebug("ExoticaValidation") << "In HLTExoticaSubAnalysis::getNamesOfObjects()";

    if (anpset.exists("recMuonLabel")) {
        _recLabels[EVTColContainer::MUON] = anpset.getParameter<edm::InputTag>("recMuonLabel");
        _genSelectorMap[EVTColContainer::MUON] = 0 ;
    }
    if (anpset.exists("recElecLabel")) {
        _recLabels[EVTColContainer::ELEC] = anpset.getParameter<edm::InputTag>("recElecLabel");
        _genSelectorMap[EVTColContainer::ELEC] = 0 ;
    }
    if (anpset.exists("recPhotonLabel")) {
        _recLabels[EVTColContainer::PHOTON] = anpset.getParameter<edm::InputTag>("recPhotonLabel");
        _genSelectorMap[EVTColContainer::PHOTON] = 0 ;
    }
    if (anpset.exists("recPFMETLabel")) {
        _recLabels[EVTColContainer::PFMET] = anpset.getParameter<edm::InputTag>("recPFMETLabel");
        _genSelectorMap[EVTColContainer::PFMET] = 0 ;
    }
    if (anpset.exists("recPFTauLabel")) {
        _recLabels[EVTColContainer::PFTAU] = anpset.getParameter<edm::InputTag>("recPFTauLabel");
        _genSelectorMap[EVTColContainer::PFTAU] = 0 ;
    }
    if (anpset.exists("recJetLabel")) {
        _recLabels[EVTColContainer::JET] = anpset.getParameter<edm::InputTag>("recJetLabel");
        _genSelectorMap[EVTColContainer::JET] = 0 ;
    }

    if (_recLabels.size() < 1) {
        edm::LogError("ExoticaValidation") << "HLTExoticaSubAnalysis::getNamesOfObjects, "
                                           << "Not included any object (recMuonLabel, recElecLabel, ...)  "
                                           << "in the analysis " << _analysisname;
        return;
    }
}

// Register consumption of objects.
// I have chosen to centralize all consumes() calls here.
void HLTExoticaSubAnalysis::registerConsumes(edm::ConsumesCollector & iC)
{
    // Register that we are getting genParticles
    _genParticleToken = iC.consumes<reco::GenParticleCollection>(_genParticleLabel);    
   
    // Register that we are getting the trigger results
    _trigResultsToken = iC.consumes<edm::TriggerResults>(_trigResultsLabel);
    // Loop over _recLabels, see what we need, and register.
    // Then save the registered token in _tokens.
    // Remember: _recLabels is a map<uint, edm::InputTag>
    // Remember: _tokens    is a map<uint, edm::EDGetToken>
    LogDebug("ExoticaValidation") << "We have got " << _recLabels.size() << "recLabels";
    for (std::map<unsigned int, edm::InputTag>::iterator it = _recLabels.begin();
         it != _recLabels.end(); ++it) {
        if (it->first == EVTColContainer::MUON) {
	    edm::EDGetTokenT<reco::MuonCollection> particularToken = iC.consumes<reco::MuonCollection>(it->second);
	    edm::EDGetToken token(particularToken);
	    _tokens[it->first] = token;
	} 
	else if (it->first == EVTColContainer::ELEC) {
            edm::EDGetTokenT<reco::GsfElectronCollection> particularToken = iC.consumes<reco::GsfElectronCollection>(it->second);
	    edm::EDGetToken token(particularToken);
	    _tokens[it->first] = token;
	} 
	else if (it->first == EVTColContainer::PHOTON) {
            edm::EDGetTokenT<reco::PhotonCollection> particularToken = iC.consumes<reco::PhotonCollection>(it->second);
	    edm::EDGetToken token(particularToken);
	    _tokens[it->first] = token;
	} 
	else if (it->first == EVTColContainer::PFMET) {
            edm::EDGetTokenT<reco::PFMETCollection> particularToken = iC.consumes<reco::PFMETCollection>(it->second);
	    edm::EDGetToken token(particularToken);
	    _tokens[it->first] = token;
	} 
	else if (it->first == EVTColContainer::PFTAU) {
            edm::EDGetTokenT<reco::PFTauCollection> particularToken = iC.consumes<reco::PFTauCollection>(it->second);
	    edm::EDGetToken token(particularToken);
	    _tokens[it->first] = token;
	} 
	else if (it->first == EVTColContainer::JET) {
            edm::EDGetTokenT<reco::PFJetCollection> particularToken = iC.consumes<reco::PFJetCollection>(it->second);
	    edm::EDGetToken token(particularToken);
	    _tokens[it->first] = token;
	} 
	else {
            edm::LogError("ExoticaValidation") << "HLTExoticaSubAnalysis::registerConsumes"
                                               << " NOT IMPLEMENTED (yet) ERROR: '" << it->second.label() << "'";
        }
    }
    
}

// Setting the collections of objects in EVTColContainer
void HLTExoticaSubAnalysis::getHandlesToObjects(const edm::Event & iEvent, EVTColContainer * col)
{
    LogDebug("ExoticaValidation") << "In HLTExoticaSubAnalysis::getHandlesToObjects()";

    if (! col->isCommonInit()) {
        // Extract the trigger results (path info, pass,...)
        edm::Handle<edm::TriggerResults> trigResults;
        iEvent.getByToken(_trigResultsToken, trigResults);
        if (trigResults.isValid()) {
            col->triggerResults = trigResults.product();
	    LogDebug("ExoticaValidation") << "Added handle to triggerResults";
        }

        // Extract the genParticles
        edm::Handle<reco::GenParticleCollection> genPart;
        iEvent.getByToken(_genParticleToken, genPart);
        if (genPart.isValid()) {
            col->genParticles = genPart.product();
	    LogDebug("ExoticaValidation") << "Added handle to genParticles";
        }
    }

    // Loop over the tokens and extract all other objects
    LogDebug("ExoticaValidation") << "We have got " << _tokens.size() << "tokens";
    for (std::map<unsigned int, edm::EDGetToken>::iterator it = _tokens.begin();
         it != _tokens.end(); ++it) {
        if (it->first == EVTColContainer::MUON) {
            edm::Handle<reco::MuonCollection> theHandle;
            iEvent.getByToken(it->second, theHandle);
            col->set(theHandle.product());
        } 
	else if (it->first == EVTColContainer::ELEC) {
            edm::Handle<reco::GsfElectronCollection> theHandle;
            iEvent.getByToken(it->second, theHandle);
            col->set(theHandle.product());
        } 
	else if (it->first == EVTColContainer::PHOTON) {
            edm::Handle<reco::PhotonCollection> theHandle;
            iEvent.getByToken(it->second, theHandle);
            col->set(theHandle.product());
        } 
	else if (it->first == EVTColContainer::PFMET) {
            edm::Handle<reco::PFMETCollection> theHandle;
            iEvent.getByToken(it->second, theHandle);
            col->set(theHandle.product());
        } 
	else if (it->first == EVTColContainer::PFTAU) {
            edm::Handle<reco::PFTauCollection> theHandle;
            iEvent.getByToken(it->second, theHandle);
            col->set(theHandle.product());
        } 
	else if (it->first == EVTColContainer::JET) {
            edm::Handle<reco::PFJetCollection> theHandle;
            iEvent.getByToken(it->second, theHandle);
            col->set(theHandle.product());
        }
	else {
            edm::LogError("ExoticaValidation") << "HLTExoticaSubAnalysis::getHandlesToObjects "
                                               << " NOT IMPLEMENTED (yet) ERROR: '" << it->first << "'";
        }
    }
}

// Booking the histograms, and putting them in DQM
void HLTExoticaSubAnalysis::bookHist(DQMStore::IBooker & iBooker,
				     const std::string & source,
                                     const std::string & objType, 
				     const std::string & variable)
{
    LogDebug("ExoticaValidation") << "In HLTExoticaSubAnalysis::bookHist()";
    std::string sourceUpper = source;
    sourceUpper[0] = std::toupper(sourceUpper[0]);
    std::string name = source + objType + variable ;
    TH1F * h = 0;
    
    if (variable.find("SumEt") != std::string::npos) {
        std::string title = "Sum ET of " + sourceUpper + " " + objType;
        const size_t nBins = _parametersTurnOn.size() - 1;
        float * edges = new float[nBins + 1];
        for (size_t i = 0; i < nBins + 1; i++) {
            edges[i] = _parametersTurnOn[i];
        }
        h = new TH1F(name.c_str(), title.c_str(), nBins, edges);
        delete[] edges;
    }
    else if (variable.find("MaxPt") != std::string::npos) {
        std::string desc = (variable == "MaxPt1") ? "Leading" : "Next-to-Leading";
        std::string title = "pT of " + desc + " " + sourceUpper + " " + objType;
        const size_t nBins = _parametersTurnOn.size() - 1;
        float * edges = new float[nBins + 1];
        for (size_t i = 0; i < nBins + 1; i++) {
            edges[i] = _parametersTurnOn[i];
        }
        h = new TH1F(name.c_str(), title.c_str(), nBins, edges);
        delete[] edges;
    } else {
        std::string symbol = (variable == "Eta") ? "#eta" : "#phi";
        std::string title  = symbol + " of " + sourceUpper + " " + objType;
        std::vector<double> params = (variable == "Eta") ? _parametersEta : _parametersPhi;
        int    nBins = (int)params[0];
        double min   = params[1];
        double max   = params[2];
        h = new TH1F(name.c_str(), title.c_str(), nBins, min, max);
    }

    h->Sumw2();
    // This is the trick, that takes a normal TH1F and puts it in in the DQM
    // machinery. Seems to be easy!
    // Updated to use the new iBooker machinery.
    _elements[name] = iBooker.book1D(name, h);
    delete h;
}

// Fill the histograms
void HLTExoticaSubAnalysis::fillHist(const std::string & source, 
				     const std::string & objType, 
				     const std::string & variable, 
				     const float & value)
{
    std::string sourceUpper = source;
    sourceUpper[0] = toupper(sourceUpper[0]);
    std::string name = source + objType + variable ;

    LogDebug("ExoticaValidation") << "In HLTExoticaSubAnalysis::fillHist() " << name << " " << value;
    _elements[name]->Fill(value);
    LogDebug("ExoticaValidation") << "In HLTExoticaSubAnalysis::fillHist() " << name << " worked";
}

// Initialize the selectors
void HLTExoticaSubAnalysis::initSelector(const unsigned int & objtype)
{

    LogDebug("ExoticaValidation") << "In HLTExoticaSubAnalysis::initSelector()";

    if (objtype == EVTColContainer::MUON && _recMuonSelector == 0) {
        _recMuonSelector = new StringCutObjectSelector<reco::Muon>(_recCut[objtype]);
    } else if (objtype == EVTColContainer::ELEC && _recElecSelector == 0) {
        _recElecSelector = new StringCutObjectSelector<reco::GsfElectron>(_recCut[objtype]);
    } else if (objtype == EVTColContainer::PHOTON && _recPhotonSelector == 0) {
        _recPhotonSelector = new StringCutObjectSelector<reco::Photon>(_recCut[objtype]);
    } else if (objtype == EVTColContainer::PFMET && _recPFMETSelector == 0) {
        _recPFMETSelector = new StringCutObjectSelector<reco::PFMET>(_recCut[objtype]);
    } else if (objtype == EVTColContainer::PFTAU && _recPFTauSelector == 0) {
        _recPFTauSelector = new StringCutObjectSelector<reco::PFTau>(_recCut[objtype]);
    } else if (objtype == EVTColContainer::JET && _recJetSelector == 0) {
        _recJetSelector = new StringCutObjectSelector<reco::PFJet>(_recCut[objtype]);
    }
    /* else
    {
    FIXME: ERROR NOT IMPLEMENTED
    }*/
}

// Insert the HLT candidates
void HLTExoticaSubAnalysis::insertCandidates(const unsigned int & objType, const EVTColContainer * cols, std::vector<reco::LeafCandidate> * matches)
{
    
    LogDebug("ExoticaValidation") << "In HLTExoticaSubAnalysis::insertCandidates()"; 
    
    if (objType == EVTColContainer::MUON) {
        for (size_t i = 0; i < cols->muons->size(); i++) {
	    LogDebug("ExoticaValidation") << "Inserting muon " << i ;
	    if (_recMuonSelector->operator()(cols->muons->at(i))) {
		reco::LeafCandidate m(0, cols->muons->at(i).p4(), cols->muons->at(i).vertex(), objType, 0, true);
		matches->push_back(m);
	    }
        }
    } else if (objType == EVTColContainer::ELEC) {
        for (size_t i = 0; i < cols->electrons->size(); i++) {
	    LogDebug("ExoticaValidation") << "Inserting electron " << i ;
            if (_recElecSelector->operator()(cols->electrons->at(i))) {
		reco::LeafCandidate m(0, cols->electrons->at(i).p4(), cols->electrons->at(i).vertex(), objType, 0, true);
		matches->push_back(m);
	    }
        }
    } else if (objType == EVTColContainer::PHOTON) {
        for (size_t i = 0; i < cols->photons->size(); i++) {
	    LogDebug("ExoticaValidation") << "Inserting photon " << i ;
            if (_recPhotonSelector->operator()(cols->photons->at(i))) {
		reco::LeafCandidate m(0, cols->photons->at(i).p4(), cols->photons->at(i).vertex(), objType, 0, true);
		matches->push_back(m);
            }
        }
    } else if (objType == EVTColContainer::PFMET) {
	/// This is a special case. Passing a PFMET* to the constructor of
	/// MatchStruct will trigger the usage of the special constructor which
	/// also sets the sumEt member.
        for (size_t i = 0; i < cols->pfMETs->size(); i++) {
	    LogDebug("ExoticaValidation") << "Inserting PFMET " << i ;
	    if (_recPFMETSelector->operator()(cols->pfMETs->at(i))) {
		reco::LeafCandidate m(0, cols->pfMETs->at(i).p4(), cols->pfMETs->at(i).vertex(), objType, 0, true);
		matches->push_back(m);
	    }
        }
    } else if (objType == EVTColContainer::PFTAU) {
        for (size_t i = 0; i < cols->pfTaus->size(); i++) {
	    LogDebug("ExoticaValidation") << "Inserting PFtau " << i ;
	    if (_recPFTauSelector->operator()(cols->pfTaus->at(i))) {
		reco::LeafCandidate m(0, cols->pfTaus->at(i).p4(), cols->pfTaus->at(i).vertex(), objType, 0, true);
		matches->push_back(m);
	    }
        }
    } else if (objType == EVTColContainer::JET) {
        for (size_t i = 0; i < cols->jets->size(); i++) {
	    LogDebug("ExoticaValidation") << "Inserting jet " << i ;
            if (_recJetSelector->operator()(cols->jets->at(i))) {
		reco::LeafCandidate m(0, cols->jets->at(i).p4(), cols->jets->at(i).vertex(), objType, 0, true);
		matches->push_back(m);
            }
        }
    }

    /* else
    {
    FIXME: ERROR NOT IMPLEMENTED
    }*/
}
