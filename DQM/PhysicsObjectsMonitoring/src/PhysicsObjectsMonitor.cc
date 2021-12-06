/** \class PhysicsObjectsMonitor
 *  Analyzer of the StandAlone muon tracks
 *
 *  \author M. Mulders - CERN <martijn.mulders@cern.ch>
 *  Based on STAMuonAnalyzer by R. Bellan - INFN Torino
 *<riccardo.bellan@cern.ch>
 */

#include "DQM/PhysicsObjectsMonitoring/interface/PhysicsObjectsMonitor.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include <FWCore/MessageLogger/interface/MessageLogger.h>

using namespace std;
using namespace edm;

/// Constructor
PhysicsObjectsMonitor::PhysicsObjectsMonitor(const ParameterSet &pset) {
  theSTAMuonLabel = pset.getUntrackedParameter<string>("StandAloneTrackCollectionLabel");
  theSeedCollectionLabel = pset.getUntrackedParameter<string>("MuonSeedCollectionLabel");
  theDataType = pset.getUntrackedParameter<string>("DataType");
  magFiledToken_ = esConsumes();
  if (theDataType != "RealData" && theDataType != "SimData")
    edm::LogInfo("PhysicsObjectsMonitor") << "Error in Data Type!!" << endl;

  if (theDataType == "SimData") {
    edm::LogInfo("PhysicsObjectsMonitor") << "Sorry! Running this package on simulation is no longer supported! ";
  }

  // set Token(-s)
  theSTAMuonToken_ =
      consumes<reco::TrackCollection>(pset.getUntrackedParameter<string>("StandAloneTrackCollectionLabel"));
}

/// Destructor
PhysicsObjectsMonitor::~PhysicsObjectsMonitor() {}

void PhysicsObjectsMonitor::bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &, edm::EventSetup const &) {
  iBooker.setCurrentFolder("PhysicsObjects/MuonReconstruction");

  hPres = iBooker.book1D("pTRes", "pT Resolution", 100, -2, 2);
  h1_Pres = iBooker.book1D("invPTRes", "1/pT Resolution", 100, -2, 2);

  charge = iBooker.book1D("charge", "track charge", 5, -2.5, 2.5);
  ptot = iBooker.book1D("ptot", "track momentum", 50, 0, 50);
  pt = iBooker.book1D("pt", "track pT", 100, 0, 50);
  px = iBooker.book1D("px ", "track px", 100, -50, 50);
  py = iBooker.book1D("py", "track py", 100, -50, 50);
  pz = iBooker.book1D("pz", "track pz", 100, -50, 50);
  Nmuon = iBooker.book1D("Nmuon", "Number of muon tracks", 11, -.5, 10.5);
  Nrechits = iBooker.book1D("Nrechits", "Number of RecHits/Segments on track", 21, -.5, 21.5);
  NDThits = iBooker.book1D("NDThits", "Number of DT Hits/Segments on track", 31, -.5, 31.5);
  NCSChits = iBooker.book1D("NCSChits", "Number of CSC Hits/Segments on track", 31, -.5, 31.5);
  NRPChits = iBooker.book1D("NRPChits", "Number of RPC hits on track", 11, -.5, 11.5);

  DTvsCSC = iBooker.book2D("DTvsCSC", "Number of DT vs CSC hits on track", 29, -.5, 28.5, 29, -.5, 28.5);
  TH2F *root_ob = DTvsCSC->getTH2F();
  root_ob->SetXTitle("Number of DT hits");
  root_ob->SetYTitle("Number of CSC hits");
}

void PhysicsObjectsMonitor::analyze(const Event &event, const EventSetup &eventSetup) {
  edm::LogInfo("PhysicsObjectsMonitor") << "Run: " << event.id().run() << " Event: " << event.id().event();
  MuonPatternRecoDumper debug;

  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> staTracks;
  event.getByToken(theSTAMuonToken_, staTracks);

  const auto &theMGField = eventSetup.getHandle(magFiledToken_);

  double recPt = 0.;
  double simPt = 0.;

  reco::TrackCollection::const_iterator staTrack;

  edm::LogInfo("PhysicsObjectsMonitor") << "Reconstructed tracks: " << staTracks->size() << endl;
  Nmuon->Fill(staTracks->size());
  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack) {
    reco::TransientTrack track(*staTrack, &*theMGField);

    int nrechits = 0;
    int nDThits = 0;
    int nCSChits = 0;
    int nRPChits = 0;

    for (trackingRecHit_iterator it = track.recHitsBegin(); it != track.recHitsEnd(); it++) {
      if ((*it)->isValid()) {
        edm::LogInfo("PhysicsObjectsMonitor") << "Analyzer:  Aha this looks like a Rechit!" << std::endl;
        if ((*it)->geographicalId().subdetId() == MuonSubdetId::DT) {
          nDThits++;
        } else if ((*it)->geographicalId().subdetId() == MuonSubdetId::CSC) {
          nCSChits++;
        } else if ((*it)->geographicalId().subdetId() == MuonSubdetId::RPC) {
          nRPChits++;
        } else {
          edm::LogInfo("PhysicsObjectsMonitor") << "This is an UNKNOWN hit !! " << std::endl;
        }
        nrechits++;
      }
    }

    Nrechits->Fill(nrechits);
    NDThits->Fill(nDThits);
    NCSChits->Fill(nCSChits);
    DTvsCSC->Fill(nDThits, nCSChits);
    NRPChits->Fill(nRPChits);

    debug.dumpFTS(track.impactPointTSCP().theState());

    recPt = track.impactPointTSCP().momentum().perp();
    edm::LogInfo("PhysicsObjectsMonitor")
        << " p: " << track.impactPointTSCP().momentum().mag() << " pT: " << recPt << endl;
    pt->Fill(recPt);
    ptot->Fill(track.impactPointTSCP().momentum().mag());
    charge->Fill(track.impactPointTSCP().charge());
    px->Fill(track.impactPointTSCP().momentum().x());
    py->Fill(track.impactPointTSCP().momentum().y());
    pz->Fill(track.impactPointTSCP().momentum().z());
  }
  edm::LogInfo("PhysicsObjectsMonitor") << "---" << endl;
  if (recPt && theDataType == "SimData") {
    hPres->Fill((recPt - simPt) / simPt);
    h1_Pres->Fill((1 / recPt - 1 / simPt) / (1 / simPt));
  }
}

DEFINE_FWK_MODULE(PhysicsObjectsMonitor);
