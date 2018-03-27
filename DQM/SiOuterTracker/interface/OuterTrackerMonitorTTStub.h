#ifndef SiOuterTracker_OuterTrackerMonitorTTStub_h
#define SiOuterTracker_OuterTrackerMonitorTTStub_h

#include <vector>
#include <memory>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"


class DQMStore;

class OuterTrackerMonitorTTStub : public DQMEDAnalyzer {

public:
  explicit OuterTrackerMonitorTTStub(const edm::ParameterSet&);
  ~OuterTrackerMonitorTTStub() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  // TTStub stacks
  // Global position of the stubs
  MonitorElement* Stub_Barrel_XY = nullptr;  //TTStub barrel y vs x
  MonitorElement* Stub_Endcap_Fw_XY = nullptr; //TTStub Forward Endcap y vs. x
  MonitorElement* Stub_Endcap_Bw_XY = nullptr; //TTStub Backward Endcap y vs. x
  MonitorElement* Stub_RZ = nullptr; // TTStub #rho vs. z

  // Number of stubs
  MonitorElement* Stub_Barrel = nullptr; //TTStub per layer
  MonitorElement* Stub_Endcap_Disc = nullptr; // TTStubs per disc
  MonitorElement* Stub_Endcap_Disc_Fw = nullptr; //TTStub per disc
  MonitorElement* Stub_Endcap_Disc_Bw = nullptr; //TTStub per disc
  MonitorElement* Stub_Endcap_Ring = nullptr; // TTStubs per ring
  MonitorElement* Stub_Endcap_Ring_Fw[5] = {nullptr, nullptr, nullptr, nullptr, nullptr}; // TTStubs per EC ring
  MonitorElement* Stub_Endcap_Ring_Bw[5] = {nullptr, nullptr, nullptr, nullptr, nullptr}; //TTStub per EC ring

  // Stub distribution
  MonitorElement* Stub_Eta = nullptr; //TTstub eta distribution
  MonitorElement* Stub_Phi = nullptr; //TTstub phi distribution
  MonitorElement* Stub_R = nullptr; //TTstub r distribution

  // STUB Displacement - offset
  MonitorElement* Stub_Barrel_W = nullptr; //TTstub Pos-Corr Displacement (layer)
  MonitorElement* Stub_Barrel_O = nullptr; // TTStub Offset (layer)
  MonitorElement* Stub_Endcap_Disc_W = nullptr; // TTstub Pos-Corr Displacement (disc)
  MonitorElement* Stub_Endcap_Disc_O = nullptr; // TTStub Offset (disc)
  MonitorElement* Stub_Endcap_Ring_W = nullptr; // TTstub Pos-Corr Displacement (EC ring)
  MonitorElement* Stub_Endcap_Ring_O = nullptr; // TTStub Offset (EC ring)
  MonitorElement* Stub_Endcap_Ring_W_Fw[5] = {nullptr, nullptr, nullptr, nullptr, nullptr}; // TTstub Pos-Corr Displacement (EC ring)
  MonitorElement* Stub_Endcap_Ring_O_Fw[5] = {nullptr, nullptr, nullptr, nullptr, nullptr}; // TTStub Offset (EC ring)
  MonitorElement* Stub_Endcap_Ring_W_Bw[5] = {nullptr, nullptr, nullptr, nullptr, nullptr}; // TTstub Pos-Corr Displacement (EC ring)
  MonitorElement* Stub_Endcap_Ring_O_Bw[5] = {nullptr, nullptr, nullptr, nullptr, nullptr}; // TTStub Offset (EC ring)

 private:
  edm::ParameterSet conf_;
  edm::EDGetTokenT<edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > > >  tagTTStubsToken_;
  std::string topFolderName_;
};
#endif
