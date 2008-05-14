#ifndef SegmentTrackAnalyzer_H
#define SegmentTrackAnalyzer_H


/** \class SegmentTrackAnalyzer
 *
 *  DQM monitoring source for segments associated to the muon track
 *
 *  $Date: 2008/05/13 14:52:39 $
 *  $Revision: 1.2 $
 *  \author G. Mila - INFN Torino
 */


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMOffline/Muon/src/MuonAnalyzerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "DQMOffline/Muon/src/SegmentsTrackAssociator.h"


class SegmentTrackAnalyzer : public MuonAnalyzerBase {
 public:

  /// Constructor
  SegmentTrackAnalyzer(const edm::ParameterSet&, MuonServiceProxy *theService);
  
  /// Destructor
  virtual ~SegmentTrackAnalyzer();
  
  /// Inizialize parameters for histo binning
  void beginJob(edm::EventSetup const& iSetup, DQMStore *dbe);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&, const reco::Track& recoTrack);


 private:
  // ----------member data ---------------------------
  
  edm::ParameterSet parameters;
  // Switch for verbosity
  std::string metname;
  // Find the segments associated to the track
  SegmentsTrackAssociator* theSegmentsAssociator;

  // the histos
  MonitorElement* hitsNotUsed;
  MonitorElement* TrackSegm;
  MonitorElement* hitStaProvenance;

 };
#endif  
