#ifndef SegmentTrackAnalyzer_H
#define SegmentTrackAnalyzer_H


/** \class SegmentTrackAnalyzer
 *
 *  DQM monitoring source for segments associated to the muon track
 *
 *  $Date: 2009/12/22 17:43:41 $
 *  $Revision: 1.8 $
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
#include "RecoMuon/TrackingTools/interface/SegmentsTrackAssociator.h"


class SegmentTrackAnalyzer : public MuonAnalyzerBase {
 public:

  /// Constructor
  SegmentTrackAnalyzer(const edm::ParameterSet&, MuonServiceProxy *theService);
  
  /// Destructor
  virtual ~SegmentTrackAnalyzer();
  
  /// Inizialize parameters for histo binning
  void beginJob(DQMStore *dbe);

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
  MonitorElement* hitsNotUsedPercentual;
  MonitorElement* TrackSegm;
  MonitorElement* hitStaProvenance;
  MonitorElement* hitTkrProvenance;
  MonitorElement* trackHitPercentualVsEta;
  MonitorElement* trackHitPercentualVsPhi;
  MonitorElement* trackHitPercentualVsPt;
  MonitorElement* dtTrackHitPercentualVsEta;
  MonitorElement* dtTrackHitPercentualVsPhi;
  MonitorElement* dtTrackHitPercentualVsPt;
  MonitorElement* cscTrackHitPercentualVsEta;
  MonitorElement* cscTrackHitPercentualVsPhi;
  MonitorElement* cscTrackHitPercentualVsPt;

 };
#endif  
