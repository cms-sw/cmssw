/** \file HLTExoticaPlotter.cc
 */


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"

#include "HLTriggerOffline/Exotica/interface/HLTExoticaPlotter.h"
#include "HLTriggerOffline/Exotica/interface/HLTExoticaSubAnalysis.h"
#include "HLTriggerOffline/Exotica/src/EVTColContainer.cc"

#include "TPRegexp.h"


#include<set>
#include<cctype>

HLTExoticaPlotter::HLTExoticaPlotter(const edm::ParameterSet & pset,
                                     const std::string & hltPath,
                                     const std::vector<unsigned int> & objectsType) :
    _hltPath(hltPath),
    _hltProcessName(pset.getParameter<std::string>("hltProcessName")),
    _objectsType(std::set<unsigned int>(objectsType.begin(), objectsType.end())),
    _nObjects(objectsType.size()),
    _parametersEta(pset.getParameter<std::vector<double> >("parametersEta")),
    _parametersPhi(pset.getParameter<std::vector<double> >("parametersPhi")),
    _parametersTurnOn(pset.getParameter<std::vector<double> >("parametersTurnOn"))
{
    LogDebug("ExoticaValidation") << "In HLTExoticaPlotter::constructor()";
}

HLTExoticaPlotter::~HLTExoticaPlotter()
{
}


void HLTExoticaPlotter::beginJob()
{
}



void HLTExoticaPlotter::plotterBookHistos(DQMStore::IBooker & iBooker,
					  const edm::Run & iRun,
					  const edm::EventSetup & iSetup)
{
    LogDebug("ExoticaValidation") << "In HLTExoticaPlotter::plotterBookHistos()";
    for (std::set<unsigned int>::iterator it = _objectsType.begin();
         it != _objectsType.end(); ++it) {
        std::vector<std::string> sources(2);
        sources[0] = "gen";
        sources[1] = "rec";

        const std::string objTypeStr = EVTColContainer::getTypeString(*it);

        for (size_t i = 0; i < sources.size(); i++) {
            std::string source = sources[i];
            bookHist(iBooker, source, objTypeStr, "Eta");
            bookHist(iBooker, source, objTypeStr, "Phi");
            bookHist(iBooker, source, objTypeStr, "MaxPt1");
            bookHist(iBooker, source, objTypeStr, "MaxPt2");
            bookHist(iBooker, source, objTypeStr, "SumEt");
        }
    }
}

void HLTExoticaPlotter::analyze(const bool & isPassTrigger,
                                const std::string & source,
                                const std::vector<reco::LeafCandidate> & matches)
{
    LogDebug("ExoticaValidation") << "In HLTExoticaPlotter::analyze()";
    if (!isPassTrigger) {
        return;
    }

    std::map<unsigned int, int> countobjects;
    // Initializing the count of the used object
    for (std::set<unsigned int>::iterator co = _objectsType.begin();
         co != _objectsType.end(); ++co) {
        countobjects[*co] = 0;
    }

    int counttotal = 0;
    const int totalobjectssize2 = 2 * countobjects.size();
    // Fill the histos if pass the trigger (just the two with higher pt)
    for (size_t j = 0; j < matches.size(); ++j) {
        // Is this object owned by this trigger? If not we are not interested...
        if (_objectsType.find(matches[j].pdgId()) == _objectsType.end()) {
            continue;
        }

        const unsigned int objType = matches[j].pdgId();
        const std::string objTypeStr = EVTColContainer::getTypeString(objType);

        float pt  =   matches[j].pt();
        float eta =   matches[j].eta();
        float phi =   matches[j].phi();
	float sumEt = 0;//matches[j].sumEt;
        this->fillHist(isPassTrigger, source, objTypeStr, "Eta", eta);
        this->fillHist(isPassTrigger, source, objTypeStr, "Phi", phi);
	this->fillHist(isPassTrigger, source, objTypeStr, "SumEt", sumEt);

        if (countobjects[objType] == 0) {
            this->fillHist(isPassTrigger, source, objTypeStr, "MaxPt1", pt);
            // Filled the high pt ...
            ++(countobjects[objType]);
            ++counttotal;
        } else if (countobjects[objType] == 1) {
            this->fillHist(isPassTrigger, source, objTypeStr, "MaxPt2", pt);
            // Filled the second high pt ...
            ++(countobjects[objType]);
            ++counttotal;
        } else {
            if (counttotal == totalobjectssize2) {
                break;
            }
        }
    }
}


void HLTExoticaPlotter::bookHist(DQMStore::IBooker & iBooker,
				 const std::string & source,
                                 const std::string & objType,
                                 const std::string & variable)
{
    LogDebug("ExoticaValidation") << "In HLTExoticaPlotter::bookHist()";
    std::string sourceUpper = source;
    sourceUpper[0] = std::toupper(sourceUpper[0]);
    std::string name = source + objType + variable + "_" + _hltPath;
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
        std::string title = "pT of " + desc + " " + sourceUpper + " " + objType + " "
                            "where event pass the " + _hltPath;
        const size_t nBins = _parametersTurnOn.size() - 1;
        float * edges = new float[nBins + 1];
        for (size_t i = 0; i < nBins + 1; i++) {
            edges[i] = _parametersTurnOn[i];
        }
        h = new TH1F(name.c_str(), title.c_str(), nBins, edges);
        delete [] edges;
    } else {
        std::string symbol = (variable == "Eta") ? "#eta" : "#phi";
        std::string title  = symbol + " of " + sourceUpper + " " + objType + " " +
                             "where event pass the " + _hltPath;
        std::vector<double> params = (variable == "Eta") ? _parametersEta : _parametersPhi;

        int    nBins = (int)params[0];
        double min   = params[1];
        double max   = params[2];
        h = new TH1F(name.c_str(), title.c_str(), nBins, min, max);
    }
    h->Sumw2();
    _elements[name] = iBooker.book1D(name, h);
    //    LogDebug("ExoticaValidation") << "                        booked histo with name " << name << "\n"
    //				  << "                        at location " << (unsigned long int)_elements[name];
    delete h;
}

void HLTExoticaPlotter::fillHist(const bool & passTrigger,
                                 const std::string & source,
                                 const std::string & objType,
                                 const std::string & variable,
                                 const float & value)
{
    std::string sourceUpper = source;
    sourceUpper[0] = toupper(sourceUpper[0]);
    std::string name = source + objType + variable + "_" + _hltPath;

    LogDebug("ExoticaValidation") << "In HLTExoticaPlotter::fillHist()" << name << " " << value;
    _elements[name]->Fill(value);
    LogDebug("ExoticaValidation") << "In HLTExoticaPlotter::fillHist()" << name << " worked";
}


