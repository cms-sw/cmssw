 /** \file DQMOffline/Trigger/HLTMuonMatchAndPlotContainer.cc
 *
 */

#include <utility>

#include "DQMOffline/Trigger/interface/HLTMuonMatchAndPlotContainer.h"

//////////////////////////////////////////////////////////////////////////////
//////// Namespaces and Typedefs /////////////////////////////////////////////

using namespace std;
using namespace edm;
using namespace reco;
using namespace trigger;

//////////////////////////////////////////////////////////////////////////////
///Container Class Members (this is what is used by the DQM module) //////////

/// Constructor
HLTMuonMatchAndPlotContainer::HLTMuonMatchAndPlotContainer(ConsumesCollector && iC, const ParameterSet & pset) 
{

  plotters_.clear();

  string hltProcessName  = pset.getParameter<string>("hltProcessName");

  ParameterSet inputTags = pset.getParameter<ParameterSet>("inputTags");

  InputTag resTag = inputTags.getParameter<InputTag>("triggerResults");
  InputTag sumTag = inputTags.getParameter<InputTag>("triggerSummary");
  resTag = InputTag(resTag.label(), resTag.instance(), hltProcessName);
  sumTag = InputTag(sumTag.label(), sumTag.instance(), hltProcessName);
  
  trigSummaryToken_ = iC.consumes<TriggerEvent>(sumTag);
  trigResultsToken_ = iC.consumes<TriggerResults>(resTag);
  
  bsToken_   = iC.consumes<BeamSpot>(inputTags.getParameter<InputTag>("beamSpot"));
  muonToken_ = iC.consumes<MuonCollection>(inputTags.getParameter<InputTag>("recoMuon"));
  pvToken_   = iC.consumes<VertexCollection>(inputTags.getParameter<InputTag>("offlinePVs"));

}


/// Add a HLTMuonMatchAndPlot for a given path
void HLTMuonMatchAndPlotContainer::addPlotter(const edm::ParameterSet &pset , std::string path,
					      std::string label, bool islastfilter)
{

  plotters_.emplace_back(pset,std::move(path),std::move(label),islastfilter);

}


void HLTMuonMatchAndPlotContainer::beginRun(DQMStore::IBooker & iBooker,
					    const edm::Run & iRun, 
					    const edm::EventSetup & iSetup)
{

  auto iter = plotters_.begin();
  auto end  = plotters_.end();

  for (; iter != end; ++iter) 
    {
      iter->beginRun(iBooker, iRun, iSetup);
    }

}


void HLTMuonMatchAndPlotContainer::endRun(const edm::Run & iRun, 
					  const edm::EventSetup & iSetup)
{

  auto iter = plotters_.begin();
  auto end  = plotters_.end();

  for (; iter != end; ++iter) 
    {
      iter->endRun(iRun, iSetup);
    }
  
}


void HLTMuonMatchAndPlotContainer::analyze(const edm::Event & iEvent, 
					   const edm::EventSetup & iSetup)
{  

  // Get objects from the event.  
  Handle<TriggerEvent> triggerSummary;
  iEvent.getByToken(trigSummaryToken_, triggerSummary);

  if(!triggerSummary.isValid()) 
  {
    LogError("HLTMuonMatchAndPlot")<<"Missing triggerSummary collection" << endl;
    return;
  }

  Handle<TriggerResults> triggerResults;
  iEvent.getByToken(trigResultsToken_, triggerResults);
  const edm::TriggerNames& trigNames = iEvent.triggerNames(*triggerResults);

  if(!triggerResults.isValid()) 
  {
    LogError("HLTMuonMatchAndPlot")<<"Missing triggerResults collection" << endl;
    return;
  }

  Handle<MuonCollection> allMuons;
  iEvent.getByToken(muonToken_, allMuons);

  if(!allMuons.isValid()) 
  {
    LogError("HLTMuonMatchAndPlot")<<"Missing muon collection " << endl;
    return;
  }

  Handle<BeamSpot> beamSpot;
  iEvent.getByToken(bsToken_, beamSpot);

  if(!beamSpot.isValid()) 
  {
    LogError("HLTMuonMatchAndPlot")<<"Missing beam spot collection " << endl;
    return;
  }

  Handle<VertexCollection> vertices;
  iEvent.getByToken(pvToken_, vertices);

  if(!vertices.isValid()) 
  {
    LogError("HLTMuonMatchAndPlot")<<"Missing vertices collection " << endl;
    return;
  }
  

  auto iter = plotters_.begin();
  auto end  = plotters_.end();

  for (; iter != end; ++iter) 
    {
      iter->analyze(allMuons, beamSpot, vertices, triggerSummary, triggerResults, trigNames);
    }
  
}

