
/** \file HLTHiggsPlotter.cc
 *  $Date: 2013/06/05 13:50:24 $
 *  $Revision: 1.8 $
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
                                 const std::vector<unsigned int> & objectsType,
                                 DQMStore * dbe) :
    _hltPath(hltPath),
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
    std::string objStr = EVTColContainer::getTypeString( *it );
    _cutMinPt[*it] = pset.getParameter<double>( std::string(objStr+"_cutMinPt").c_str() );
    _cutMaxEta[*it] = pset.getParameter<double>( std::string(objStr+"_cutMaxEta").c_str() );
  }
}

HLTHiggsPlotter::~HLTHiggsPlotter()
{
}


void HLTHiggsPlotter::beginJob() 
{
}



void HLTHiggsPlotter::beginRun(const edm::Run & iRun,
                               const edm::EventSetup & iSetup)
{
  for (std::set<unsigned int>::iterator it = _objectsType.begin(); 
      it != _objectsType.end(); ++it)
  {
    std::vector<std::string> sources(2);
    sources[0] = "gen";
    sources[1] = "rec";

    const std::string objTypeStr = EVTColContainer::getTypeString(*it);
	  
    for (size_t i = 0; i < sources.size(); i++) 
    {
      std::string source = sources[i];
      bookHist(source, objTypeStr, "Eta");
      bookHist(source, objTypeStr, "Phi");
      bookHist(source, objTypeStr, "MaxPt1");
      bookHist(source, objTypeStr, "MaxPt2");
    }
  }
}

void HLTHiggsPlotter::analyze(const bool & isPassTrigger,
                              const std::string & source,
                              const std::vector<MatchStruct> & matches)
{
  if ( !isPassTrigger )
  {
    return;
  }
  std::map<unsigned int,int> countobjects;
  // Initializing the count of the used object
  for(std::set<unsigned int>::iterator co = _objectsType.begin();
      co != _objectsType.end(); ++co)
  {
    countobjects[*co] = 0;
  }
	
  int counttotal = 0;
  const int totalobjectssize2 = 2*countobjects.size();
  // Fill the histos if pass the trigger (just the two with higher pt)
  for (size_t j = 0; j < matches.size(); ++j)
  {
    // Is this object owned by this trigger? If not we are not interested...
    if ( _objectsType.find( matches[j].objType) == _objectsType.end() )
    {
      continue;
    }

    const unsigned int objType = matches[j].objType;
    const std::string objTypeStr = EVTColContainer::getTypeString(matches[j].objType);
		
    float pt  = matches[j].pt;
    float eta = matches[j].eta;
    float phi = matches[j].phi;
    this->fillHist(isPassTrigger,source,objTypeStr,"Eta",eta);
    this->fillHist(isPassTrigger,source,objTypeStr,"Phi",phi);
    if ( countobjects[objType] == 0 )
    {
      this->fillHist(isPassTrigger,source,objTypeStr,"MaxPt1",pt);
      // Filled the high pt ...
      ++(countobjects[objType]);
      ++counttotal;
    }
    else if ( countobjects[objType] == 1 )
    {
      this->fillHist(isPassTrigger,source,objTypeStr,"MaxPt2",pt);
      // Filled the second high pt ...
      ++(countobjects[objType]);
      ++counttotal;
    }
    else
    {
      if ( counttotal == totalobjectssize2 ) 
      {
        break;
      }
    }				
  }
}


void HLTHiggsPlotter::bookHist(const std::string & source, 
                               const std::string & objType,
                               const std::string & variable)
{
  std::string sourceUpper = source; 
  sourceUpper[0] = std::toupper(sourceUpper[0]);
  std::string name = source + objType + variable + "_" + _hltPath;
  TH1F * h = 0;

  if (variable.find("MaxPt") != std::string::npos) 
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
    delete [] edges;
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

void HLTHiggsPlotter::fillHist(const bool & passTrigger,
                               const std::string & source, 
                               const std::string & objType,
                               const std::string & variable,
                               const float & value )
{
  std::string sourceUpper = source; 
  sourceUpper[0] = toupper(sourceUpper[0]);
  std::string name = source + objType + variable + "_" + _hltPath;

  _elements[name]->Fill(value);
}


