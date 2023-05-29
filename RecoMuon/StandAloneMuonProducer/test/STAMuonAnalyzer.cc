/** \class STAMuonAnalyzer
 *  Analyzer of the StandAlone muon tracks
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "RecoMuon/StandAloneMuonProducer/test/STAMuonAnalyzer.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

using namespace std;
using namespace edm;

/// Constructor
STAMuonAnalyzer::STAMuonAnalyzer(const ParameterSet& pset) {
  theSTAMuonLabel = pset.getUntrackedParameter<string>("StandAloneTrackCollectionLabel");
  theSeedCollectionLabel = pset.getUntrackedParameter<string>("MuonSeedCollectionLabel");

  theRootFileName = pset.getUntrackedParameter<string>("rootFileName");

  theDataType = pset.getUntrackedParameter<string>("DataType");

  if (theDataType != "RealData" && theDataType != "SimData")
    cout << "Error in Data Type!!" << endl;

  numberOfSimTracks = 0;
  numberOfRecTracks = 0;

  theFieldToken = esConsumes();
  theGeomToken = esConsumes();
}

/// Destructor
STAMuonAnalyzer::~STAMuonAnalyzer() {}

void STAMuonAnalyzer::beginJob() {
  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theFile->cd();

  hPtRec = new TH1F("pTRec", "p_{T}^{rec}", 250, 0, 120);
  hPtSim = new TH1F("pTSim", "p_{T}^{gen} ", 250, 0, 120);

  hPTDiff = new TH1F("pTDiff", "p_{T}^{rec} - p_{T}^{gen} ", 250, -120, 120);
  hPTDiff2 = new TH1F("pTDiff2", "p_{T}^{rec} - p_{T}^{gen} ", 250, -120, 120);

  hPTDiffvsEta = new TH2F("PTDiffvsEta", "p_{T}^{rec} - p_{T}^{gen} VS #eta", 100, -2.5, 2.5, 250, -120, 120);
  hPTDiffvsPhi = new TH2F("PTDiffvsPhi", "p_{T}^{rec} - p_{T}^{gen} VS #phi", 100, -6, 6, 250, -120, 120);

  hPres = new TH1F("pTRes", "pT Resolution", 100, -2, 2);
  h1_Pres = new TH1F("invPTRes", "1/pT Resolution", 100, -2, 2);
}

void STAMuonAnalyzer::endJob() {
  if (theDataType == "SimData") {
    cout << endl << endl << "Number of Sim tracks: " << numberOfSimTracks << endl;
  }

  cout << "Number of Reco tracks: " << numberOfRecTracks << endl << endl;

  // Write the histos to file
  theFile->cd();
  hPtRec->Write();
  hPtSim->Write();
  hPres->Write();
  h1_Pres->Write();
  hPTDiff->Write();
  hPTDiff2->Write();
  hPTDiffvsEta->Write();
  hPTDiffvsPhi->Write();
  theFile->Close();
}

void STAMuonAnalyzer::analyze(const Event& event, const EventSetup& eventSetup) {
  cout << "Run: " << event.id().run() << " Event: " << event.id().event() << endl;
  MuonPatternRecoDumper debug;

  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> staTracks;
  event.getByLabel(theSTAMuonLabel, staTracks);

  ESHandle<MagneticField> theMGField = eventSetup.getHandle(theFieldToken);
  ESHandle<GlobalTrackingGeometry> theTrackingGeometry = eventSetup.getHandle(theGeomToken);

  double recPt = 0.;
  double simPt = 0.;

  // Get the SimTrack collection from the event
  if (theDataType == "SimData") {
    Handle<SimTrackContainer> simTracks;
    event.getByLabel("g4SimHits", simTracks);

    numberOfRecTracks += staTracks->size();

    SimTrackContainer::const_iterator simTrack;

    cout << "Simulated tracks: " << endl;
    for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack) {
      if (abs((*simTrack).type()) == 13) {
        cout << "Sim pT: " << (*simTrack).momentum().pt() << endl;
        simPt = (*simTrack).momentum().pt();
        cout << "Sim Eta: " << (*simTrack).momentum().eta() << endl;
        numberOfSimTracks++;
      }
    }
    cout << endl;
  }

  reco::TrackCollection::const_iterator staTrack;

  cout << "Reconstructed tracks: " << staTracks->size() << endl;

  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack) {
    reco::TransientTrack track(*staTrack, &*theMGField, theTrackingGeometry);

    cout << debug.dumpFTS(track.impactPointTSCP().theState());

    recPt = track.impactPointTSCP().momentum().perp();
    cout << " p: " << track.impactPointTSCP().momentum().mag() << " pT: " << recPt << endl;
    cout << " chi2: " << track.chi2() << endl;

    hPtRec->Fill(recPt);

    TrajectoryStateOnSurface innerTSOS = track.innermostMeasurementState();
    cout << "Inner TSOS:" << endl;
    cout << debug.dumpTSOS(innerTSOS);
    cout << " p: " << innerTSOS.globalMomentum().mag() << " pT: " << innerTSOS.globalMomentum().perp() << endl;

    trackingRecHit_iterator rhbegin = staTrack->recHitsBegin();
    trackingRecHit_iterator rhend = staTrack->recHitsEnd();

    cout << "RecHits:" << endl;
    for (trackingRecHit_iterator recHit = rhbegin; recHit != rhend; ++recHit) {
      const GeomDet* geomDet = theTrackingGeometry->idToDet((*recHit)->geographicalId());
      double r = geomDet->surface().position().perp();
      double z = geomDet->toGlobal((*recHit)->localPosition()).z();
      cout << "r: " << r << " z: " << z << endl;
    }

    if (recPt && theDataType == "SimData") {
      hPres->Fill((recPt - simPt) / simPt);
      hPtSim->Fill(simPt);

      hPTDiff->Fill(recPt - simPt);

      //      hPTDiff2->Fill(track.innermostMeasurementState().globalMomentum().perp()-simPt);
      hPTDiffvsEta->Fill(track.impactPointTSCP().position().eta(), recPt - simPt);
      hPTDiffvsPhi->Fill(track.impactPointTSCP().position().phi(), recPt - simPt);

      if (((recPt - simPt) / simPt) <= -0.4)
        cout << "Out of Res: " << (recPt - simPt) / simPt << endl;
      h1_Pres->Fill((1 / recPt - 1 / simPt) / (1 / simPt));
    }
  }
  cout << "---" << endl;
}

DEFINE_FWK_MODULE(STAMuonAnalyzer);
