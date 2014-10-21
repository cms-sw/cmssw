#ifndef DQMOffline_Trigger_HLTMuonMatchAndPlotContainer_H
#define DQMOffline_Trigger_HLTMuonMatchAndPlotContainer_H

/** \class HLTMuonMatchAndPlot
 *  Contanier class to handle vector of reconstructed muons matched to 
 *  HLT objects used to plot efficiencies.
 *
 *  Note that this is not a true EDAnalyzer; rather, the intent is that one
 *  EDAnalyzer would call an instance  of HLTMuonMatchAndPlotContainer
 *
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLTOfflinePerformance
 *
 *  
 *  \author  C. Battilana
 */

// Base Class Headers

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DQMOffline/Trigger/interface/HLTMuonMatchAndPlot.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/deltaR.h"

#include<vector>
#include<string>


//////////////////////////////////////////////////////////////////////////////
///Container Class Definition (this is what is used by the DQM module) ///////

class HLTMuonMatchAndPlotContainer 
{

 public:

  /// Constructor
  HLTMuonMatchAndPlotContainer(edm::ConsumesCollector &&, const edm::ParameterSet &);

  /// Destructor
  ~HLTMuonMatchAndPlotContainer() { plotters_.clear(); };

  /// Add a HLTMuonMatchAndPlot for a given path
  void addPlotter(const edm::ParameterSet &, std::string,
		  const std::vector<std::string>&);

  // Analyzer Methods
  void beginRun(DQMStore::IBooker &, const edm::Run &, const edm::EventSetup &);
  void analyze(const edm::Event &, const edm::EventSetup &);
  void endRun(const edm::Run &, const edm::EventSetup &);

 private:

  std::vector<HLTMuonMatchAndPlot> plotters_;

  edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  edm::EDGetTokenT<reco::MuonCollection> muonToken_;
  edm::EDGetTokenT<reco::VertexCollection> pvToken_;

  edm::EDGetTokenT<trigger::TriggerEvent> trigSummaryToken_;
  edm::EDGetTokenT<edm::TriggerResults>   trigResultsToken_;
  
};

#endif
 

