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
  ~OuterTrackerMonitorTTTrack();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
 
  
  MonitorElement* Track_N = 0;
  MonitorElement* Track_NStubs = 0;
  
  /// Low-quality TTTracks (made from less than X TTStubs)
  MonitorElement* Track_LQ_N = 0;
  MonitorElement* Track_LQ_Pt = 0;
  MonitorElement* Track_LQ_Eta = 0;
  MonitorElement* Track_LQ_Phi = 0;
  MonitorElement* Track_LQ_VtxZ0 = 0;
  MonitorElement* Track_LQ_Chi2 = 0;
  MonitorElement* Track_LQ_Chi2Red = 0;
  MonitorElement* Track_LQ_Chi2_NStubs = 0;
  MonitorElement* Track_LQ_Chi2Red_NStubs = 0;
  
  /// High-quality TTTracks (made from at least X TTStubs)
  MonitorElement* Track_HQ_N = 0;
  MonitorElement* Track_HQ_Pt = 0;
  MonitorElement* Track_HQ_Eta = 0;
  MonitorElement* Track_HQ_Phi = 0;
  MonitorElement* Track_HQ_VtxZ0 = 0;
  MonitorElement* Track_HQ_Chi2 = 0;
  MonitorElement* Track_HQ_Chi2Red = 0;
  MonitorElement* Track_HQ_Chi2_NStubs = 0;
  MonitorElement* Track_HQ_Chi2Red_NStubs = 0;


 private:
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;
  //edm::EDGetTokenT<edmNew::DetSetVector< TTTrack< Ref_Phase2TrackerDigi_ > > >  tagTTTracksToken_;

  std::string topFolderName_;
  unsigned int HQDelim_;
};
#endif
