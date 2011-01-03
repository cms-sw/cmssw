#ifndef DQMOffline_Trigger_HLTMuonMatchAndPlot_H
#define DQMOffline_Trigger_HLTMuonMatchAndPlot_H

/** \class HLTMuonMatchAndPlot
 *  Match reconstructed muons to HLT objects and plot efficiencies.
 *
 *  Note that this is not a true EDAnalyzer; rather, the intent is that one
 *  EDAnalyzer would call a separate instantiation of HLTMuonMatchAndPlot
 *  for each HLT path under consideration.
 *
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLTOfflinePerformance
 *
 *  $Date: 2010/07/21 04:23:22 $
 *  $Revision: 1.13 $
 *  
 *  \author  J. Slaunwhite, Jeff Klukas
 */

// Base Class Headers

//#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
//#include "DataFormats/Common/interface/RefToBase.h"
//#include "DataFormats/TrackReco/interface/Track.h"
//#include "DataFormats/Candidate/interface/Candidate.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/MuonReco/interface/Muon.h"
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
//////// Class Definition ////////////////////////////////////////////////////

class HLTMuonMatchAndPlot {

 public:

  /// Constructor
  HLTMuonMatchAndPlot(const edm::ParameterSet &, std::string,
                      std::vector<std::string>);

  // Analyzer Methods
  void beginRun(const edm::Run &, const edm::EventSetup &);
  void analyze(const edm::Event &, const edm::EventSetup &);
  void endRun(const edm::Run &, const edm::EventSetup &);

  // Helper Methods
  void fillEdges(size_t & nBins, float * & edges, std::vector<double> binning);
  template <class T> void 
    fillMapFromPSet(std::map<std::string, T> &, edm::ParameterSet, std::string);
  template <class T1, class T2> std::vector<size_t> 
    matchByDeltaR(const std::vector<T1> &, const std::vector<T2> &, 
                  double maxDeltaR = NOMATCH);

 private:

  // Internal Methods
  void book1D(std::string, std::string, std::string);
  void book2D(std::string, std::string, std::string, std::string);
  reco::MuonCollection selectedMuons(
    const reco::MuonCollection &,
    const reco::BeamSpot &,
    const edm::ParameterSet &);
  trigger::TriggerObjectCollection selectedTriggerObjects(
    const trigger::TriggerObjectCollection &,
    const trigger::TriggerEvent &,
    const edm::ParameterSet &);
 
  // Input from Configuration File
  std::string hltProcessName_;
  std::string destination_;
  std::map<std::string, edm::InputTag> inputTags_;
  std::map<std::string, std::vector<double> > binParams_;
  std::map<std::string, double > deltaRCuts_;
  edm::ParameterSet targetParams_;
  edm::ParameterSet probeParams_;

  // Member Variables
  std::string hltPath_;
  std::vector<std::string> moduleLabels_;
  DQMStore * dbe_;
  std::map<std::string, MonitorElement *> hists_;

};

#endif
