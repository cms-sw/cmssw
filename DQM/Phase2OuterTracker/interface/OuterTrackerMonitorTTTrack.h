#ifndef Phase2OuterTracker_OuterTrackerMonitorTTTrack_h
#define Phase2OuterTracker_OuterTrackerMonitorTTTrack_h

#include <vector>
#include <memory>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"


class DQMStore;

class OuterTrackerMonitorTTTrack : public edm::EDAnalyzer {

public:
  explicit OuterTrackerMonitorTTTrack(const edm::ParameterSet&);
  ~OuterTrackerMonitorTTTrack() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override ;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
 
  
  MonitorElement* Track_N = nullptr;
  MonitorElement* Track_NStubs = nullptr;
  
  /// Low-quality TTTracks (made from less than X TTStubs)
  MonitorElement* Track_LQ_N = nullptr;
  MonitorElement* Track_LQ_Pt = nullptr;
  MonitorElement* Track_LQ_Eta = nullptr;
  MonitorElement* Track_LQ_Phi = nullptr;
  MonitorElement* Track_LQ_VtxZ0 = nullptr;
  MonitorElement* Track_LQ_Chi2 = nullptr;
  MonitorElement* Track_LQ_Chi2Red = nullptr;
  MonitorElement* Track_LQ_Chi2_NStubs = nullptr;
  MonitorElement* Track_LQ_Chi2Red_NStubs = nullptr;
  
  /// High-quality TTTracks (made from at least X TTStubs)
  MonitorElement* Track_HQ_N = nullptr;
  MonitorElement* Track_HQ_Pt = nullptr;
  MonitorElement* Track_HQ_Eta = nullptr;
  MonitorElement* Track_HQ_Phi = nullptr;
  MonitorElement* Track_HQ_VtxZ0 = nullptr;
  MonitorElement* Track_HQ_Chi2 = nullptr;
  MonitorElement* Track_HQ_Chi2Red = nullptr;
  MonitorElement* Track_HQ_Chi2_NStubs = nullptr;
  MonitorElement* Track_HQ_Chi2Red_NStubs = nullptr;


 private:
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;
  //edm::EDGetTokenT<edmNew::DetSetVector< TTTrack< Ref_Phase2TrackerDigi_ > > >  tagTTTracksToken_;

  std::string topFolderName_;
  unsigned int HQDelim_;
};
#endif
