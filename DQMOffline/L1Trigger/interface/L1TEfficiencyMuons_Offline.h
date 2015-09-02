#ifndef DQMOFFLINE_L1TRIGGER_L1TEFFICIENCYMUON_OFFLINE_H
#define DQMOFFLINE_L1TRIGGER_L1TEFFICIENCYMUON_OFFLINE_H

/**
 * \file L1TEfficiencyMuons.h
 *
 * \author J. Pela, C. Battilana
 *
 */

// system include files
#include <memory>
#include <unistd.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "TRegexp.h"
#include "TString.h"

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>


//
// helper class to manage GMT-Muon pariring
//

class MuonGmtPair {

 public :

  MuonGmtPair(const reco::Muon *muon, const L1MuGMTExtendedCand *gmt) : 
    m_muon(muon), m_gmt(gmt), m_eta(999.), m_phi_bar(999.), m_phi_end(999.) { };
    
  MuonGmtPair(const MuonGmtPair& muonGmtPair);

  ~MuonGmtPair() { };

  double dR();

  double eta() const { return m_eta; };
  double phi() const { return fabs(m_eta)< 1.04 ? m_phi_bar : m_phi_end; };
  double pt()  const { return m_muon->isGlobalMuon() ? m_muon->globalTrack()->pt() : -1; };
  
  double gmtPt() const { return m_gmt ? m_gmt->ptValue() : -1.; };

  void propagate(edm::ESHandle<MagneticField> bField,
		 edm::ESHandle<Propagator> propagatorAlong,
		 edm::ESHandle<Propagator> propagatorOpposite);

private :

  // propagation private members
  TrajectoryStateOnSurface cylExtrapTrkSam(reco::TrackRef track, double rho);
  TrajectoryStateOnSurface surfExtrapTrkSam(reco::TrackRef track, double z);
  FreeTrajectoryState freeTrajStateMuon(reco::TrackRef track);

private :

  const reco::Muon *m_muon;
  const L1MuGMTExtendedCand *m_gmt;

  edm::ESHandle<MagneticField> m_BField;
  edm::ESHandle<Propagator> m_propagatorAlong;
  edm::ESHandle<Propagator> m_propagatorOpposite;

  double m_eta;
  double m_phi_bar;
  double m_phi_end;

};

//
// DQM class declaration
//

class L1TEfficiencyMuons_Offline : public DQMEDAnalyzer {
  
public:
  
  L1TEfficiencyMuons_Offline(const edm::ParameterSet& ps);   // Constructor
  virtual ~L1TEfficiencyMuons_Offline();                     // Destructor
  
protected:
  
   // Luminosity Block
  virtual void beginLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c);
  virtual void dqmEndLuminosityBlock  (edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c);
  virtual void dqmBeginRun(const edm::Run& run, const edm::EventSetup& iSetup);
  virtual void bookControlHistos(DQMStore::IBooker &);
  virtual void bookEfficiencyHistos(DQMStore::IBooker &ibooker, int ptCut);
  virtual void bookHistograms(DQMStore::IBooker &ibooker, const edm::Run& run, const edm::EventSetup& iSetup) override;
  //virtual void analyze (const edm::Event& e, const edm::EventSetup& c);

private:

  void analyze (const edm::Event& e, const edm::EventSetup& c);

  // Helper Functions
  const reco::Vertex getPrimaryVertex(edm::Handle<reco::VertexCollection> & vertex,edm::Handle<reco::BeamSpot> & beamSpot);
  bool matchHlt(edm::Handle<trigger::TriggerEvent>  & triggerEvent, 
		const reco::Muon * mu);

  // Cut and Matching
  void getMuonGmtPairs(edm::Handle<L1MuGMTReadoutCollection> & gmtCands);
  void getTightMuons(edm::Handle<reco::MuonCollection> & muons, const reco::Vertex & vertex);
  void getProbeMuons(edm::Handle<edm::TriggerResults> & trigResults,edm::Handle<trigger::TriggerEvent> & trigEvent);  
  
private:
  
  bool  m_verbose;

  HLTConfigProvider m_hltConfig;

  edm::ESHandle<MagneticField> m_BField;
  edm::ESHandle<Propagator> m_propagatorAlong;
  edm::ESHandle<Propagator> m_propagatorOpposite;

  // histos
  std::map<int, std::map<std::string, MonitorElement*> > m_EfficiencyHistos;
  std::map<std::string, MonitorElement*> m_ControlHistos;

  // helper variables
  std::vector<const reco::Muon*>  m_TightMuons;
  std::vector<const reco::Muon*>  m_ProbeMuons;
  std::vector<MuonGmtPair>  m_MuonGmtPairs;  
  
  // config params
  std::vector<int> m_GmtPtCuts;

  edm::EDGetTokenT<reco::MuonCollection> m_MuonInputTag;
  edm::EDGetTokenT<L1MuGMTReadoutCollection> m_GmtInputTag;

  edm::EDGetTokenT<reco::VertexCollection> m_VtxInputTag;
  edm::EDGetTokenT<reco::BeamSpot> m_BsInputTag;

  edm::EDGetTokenT<trigger::TriggerEvent> m_trigInputTag;
  std::string m_trigProcess;
  edm::EDGetTokenT<edm::TriggerResults> m_trigProcess_token;
  std::vector<std::string> m_trigNames;
  std::vector<int> m_trigIndices;

  float m_MaxMuonEta;
  float m_MaxGmtMuonDR;
  float m_MaxHltMuonDR;
  // CB ignored at present
  // float m_MinMuonDR;
  
};

#endif
