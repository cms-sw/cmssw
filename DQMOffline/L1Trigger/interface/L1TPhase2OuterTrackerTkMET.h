#ifndef L1TPhase2_OuterTrackerTkMET_h
#define L1TPhase2_OuterTrackerTkMET_h

#include <vector>
#include <memory>
#include <string>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
// #include "DataFormats/L1TVertex/interface/Vertex.h"
#include "DataFormats/L1Trigger/interface/VertexWord.h"

class DQMStore;
class L1TPhase2OuterTrackerTkMET : public DQMEDAnalyzer {
public:
  explicit L1TPhase2OuterTrackerTkMET(const edm::ParameterSet&);
  ~L1TPhase2OuterTrackerTkMET() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  MonitorElement* Track_N = nullptr;         // Number of tracks per event
  MonitorElement* Track_Pt = nullptr;        // pT distrubtion for tracks
  MonitorElement* Track_Eta = nullptr;       // eta distrubtion for tracks
  MonitorElement* Track_VtxZ = nullptr;      // z0 distrubtion for tracks
  MonitorElement* Track_Chi2Dof = nullptr;   // chi2 distrubtion for tracks
  MonitorElement* Track_BendChi2 = nullptr;  // bend chi2 distrubtion for tracks
  MonitorElement* Track_NStubs = nullptr;    // nstubs distrubtion for tracks
  MonitorElement* Track_NPSstubs = nullptr;  // nPS stubs distrubtion for tracks

  MonitorElement* TkMET_QualityCuts = nullptr;  //Matches the quality cuts in the producer
  MonitorElement* TkMET_PU = nullptr;

private:
  edm::ParameterSet conf_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> m_topoToken;
  edm::EDGetTokenT<l1t::VertexWordCollection> pvToken;
  edm::EDGetTokenT<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > > ttTrackToken_;

  float maxZ0;   // in cm
  float DeltaZ;  // in cm
  float maxEta;
  float chi2dofMax;
  float bendchi2Max;
  float minPt;  // in GeV
  int nStubsmin;
  int nStubsPSmin;   // minimum number of stubs in PS modules
  float maxPt;       // in GeV
  int HighPtTracks;  // saturate or truncate

  std::string topFolderName_;
};
#endif
