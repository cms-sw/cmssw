#ifndef BPhysicsOniaDQM_H
#define BPhysicsOniaDQM_H

/** \class BPhysicsOniaDQM
 *
 *  DQM offline for quarkonia
 *
 *  \author S. Bolognesi, Eric - CERN
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include <string>
#include <cmath>
#include <map>

class DQMStore;
class MonitorElement;

class BPhysicsOniaDQM : public DQMEDAnalyzer {
 public:
  /// Constructor
  BPhysicsOniaDQM(const edm::ParameterSet&);

  /// Destructor
  virtual ~BPhysicsOniaDQM();

  void bookHistograms(DQMStore::IBooker&, edm::Run const&,
                      edm::EventSetup const&) override;
  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&);

 private:
  float computeMass(const math::XYZVector& vec1, const math::XYZVector& vec2);
  bool isMuonInAccept(const reco::Muon& recoMu);
  bool selGlobalMuon(const reco::Muon& recoMu);
  bool selTrackerMuon(const reco::Muon& recoMu);

  // ----------member data ---------------------------
  edm::EDGetTokenT<reco::VertexCollection> vertex_;
  // Muon Label
  edm::EDGetTokenT<reco::MuonCollection> theMuonCollectionLabel_;
  edm::EDGetTokenT<LumiSummary> lumiSummaryToken_;

  // Switch for verbosity
  std::string metname;

  // The histos
  MonitorElement* diMuonMass_global;
  MonitorElement* diMuonMass_tracker;
  MonitorElement* diMuonMass_standalone;
  MonitorElement* global_background;
  MonitorElement* tracker_background;
  MonitorElement* standalone_background;

  MonitorElement* glbSigCut;
  MonitorElement* glbSigNoCut;
  MonitorElement* glbBkgNoCut;
  MonitorElement* staSigCut;
  MonitorElement* staSigNoCut;
  MonitorElement* staBkgNoCut;
  MonitorElement* trkSigCut;
  MonitorElement* trkSigNoCut;
  MonitorElement* trkBkgNoCut;

  math::XYZPoint RefVtx;
};
#endif
