#ifndef DQMOffline_Trigger_HLTMuonMatchAndPlot_H
#define DQMOffline_Trigger_HLTMuonMatchAndPlot_H

/** \class HLTMuonMatchAndPlot
 *  Match reconstructed muons to HLT objects and plot efficiencies.
 *
 *  Note that this is not a true EDAnalyzer;
 *
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLTOfflinePerformance
 *
 *  
 *  \author  J. Slaunwhite, Jeff Klukas
 */

// Base Class Headers

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <vector>
#include "TFile.h"
#include "TNtuple.h"
#include "TString.h"
#include "TPRegexp.h"


//////////////////////////////////////////////////////////////////////////////
//////// Typedefs and Constants //////////////////////////////////////////////

typedef math::XYZTLorentzVector LorentzVector;

const double NOMATCH = 999.;
const std::string EFFICIENCY_SUFFIXES[2] = {"denom", "numer"};


//////////////////////////////////////////////////////////////////////////////
//////// HLTMuonMatchAndPlot Class Definition ////////////////////////////////

class HLTMuonMatchAndPlot 
{

 public:

  /// Constructor
  HLTMuonMatchAndPlot(const edm::ParameterSet &, std::string,
                      const std::vector<std::string>&);

  // Analyzer Methods
  void beginRun(DQMStore::IBooker &, const edm::Run &, const edm::EventSetup &);
  void analyze(edm::Handle<reco::MuonCollection> &, edm::Handle<reco::BeamSpot> &, 
	       edm::Handle<reco::VertexCollection> &, edm::Handle<trigger::TriggerEvent> &, 
	       edm::Handle<edm::TriggerResults> &);
  void endRun(const edm::Run &, const edm::EventSetup &);

  // Helper Methods
  void fillEdges(size_t & nBins, float * & edges, const std::vector<double>& binning);
  template <class T> void 
    fillMapFromPSet(std::map<std::string, T> &, const edm::ParameterSet&, std::string);
  template <class T1, class T2> std::vector<size_t> 
    matchByDeltaR(const std::vector<T1> &, const std::vector<T2> &, 
                  const double maxDeltaR = NOMATCH);
  
 private:

  // Internal Methods
  void book1D(DQMStore::IBooker &, std::string, std::string, std::string);
  void book2D(DQMStore::IBooker &, std::string, std::string, std::string, std::string);
  reco::MuonCollection selectedMuons(
    const reco::MuonCollection &,
    const reco::BeamSpot &,
    bool,    
    const StringCutObjectSelector<reco::Muon> &,
    double, double);

  trigger::TriggerObjectCollection selectedTriggerObjects(
    const trigger::TriggerObjectCollection &,
    const trigger::TriggerEvent &,
    bool hasTriggerCuts,
    const StringCutObjectSelector<trigger::TriggerObject> triggerSelector);
 
  // Input from Configuration File
  std::string hltProcessName_;
  std::string destination_;
  std::vector<std::string> requiredTriggers_;
  std::map<std::string, std::vector<double> > binParams_;
  std::map<std::string, double> plotCuts_;
  edm::ParameterSet targetParams_;
  edm::ParameterSet probeParams_;

  // Member Variables
  std::string triggerLevel_;
  unsigned int cutMinPt_;
  std::string hltPath_;
  std::vector<std::string> moduleLabels_;
  std::map<std::string, MonitorElement *> hists_;
  
  // Selectors
  bool hasTargetRecoCuts;                                                                                                                                                                                                                                                    
  bool hasProbeRecoCuts;
    
  StringCutObjectSelector<reco::Muon> targetMuonSelector_;
  double targetZ0Cut_; 
  double targetD0Cut_;
  StringCutObjectSelector<reco::Muon> probeMuonSelector_;
  double probeZ0Cut_; 
  double probeD0Cut_;

  StringCutObjectSelector<trigger::TriggerObject> triggerSelector_;
  bool hasTriggerCuts_;

};

#endif
