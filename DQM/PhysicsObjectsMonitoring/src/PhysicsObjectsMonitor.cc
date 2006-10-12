/** \class PhysicsObjectsMonitor
 *  Analyzer of the StandAlone muon tracks
 *
 *  $Date: 2006/08/02 08:08:30 $
 *  $Revision: 1.2 $
 *  \author M. Mulders - CERN <martijn.mulders@cern.ch>
 *  Based on STAMuonAnalyzer by R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "DQM/PhysicsObjectsMonitoring/interface/PhysicsObjectsMonitor.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

using namespace std;
using namespace edm;

/// Constructor
PhysicsObjectsMonitor::PhysicsObjectsMonitor(const ParameterSet& pset){
  theSTAMuonLabel = pset.getUntrackedParameter<string>("StandAloneTrackCollectionLabel");
  theSeedCollectionLabel = pset.getUntrackedParameter<string>("MuonSeedCollectionLabel");

  theRootFileName = pset.getUntrackedParameter<string>("rootFileName");

  theDataType = pset.getUntrackedParameter<string>("DataType");
  
  if(theDataType != "RealData" && theDataType != "SimData")
    cout<<"Error in Data Type!!"<<endl;

  numberOfSimTracks=0;
  numberOfRecTracks=0;
}

/// Destructor
PhysicsObjectsMonitor::~PhysicsObjectsMonitor(){
}

void PhysicsObjectsMonitor::beginJob(const EventSetup& eventSetup){
  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theFile->cd();

  hPres = new TH1F("pTRes","pT Resolution",100,-2,2);
  h1_Pres = new TH1F("invPTRes","1/pT Resolution",100,-2,2);

  charge = new TH1F("charge","track charge",5,-2.5,2.5);
  ptot = new TH1F("ptot","track momentum",50,0,50);
  pt = new TH1F("pt","track pT",100,0,50);
  px = new TH1F("px ","track px",100,-50,50);
  py = new TH1F("py","track py",100,-50,50);
  pz = new TH1F("pz","track pz",100,-50,50);
  Nmuon = new TH1F("Nmuon","Number of muon tracks",11,-.5,10.5);
  Nrechits = new TH1F("Nrechits","Number of RecHits/Segments on track",21,-.5,21.5);
  NDThits = new TH1F("NDThits","Number of DT Hits/Segments on track",31,-.5,31.5);
  NCSChits = new TH1F("NCSChits","Number of CSC Hits/Segments on track",31,-.5,31.5);
  DTvsCSC = new TH2F("DTvsCSC","Number of DT vs CSC hits on track",29,-.5,28.5,29,-.5,28.5);
}

void PhysicsObjectsMonitor::endJob(){
  if(theDataType == "SimData"){
    cout << endl << endl << "Number of Sim tracks: " << numberOfSimTracks << endl;
  }

  cout << "Number of Reco tracks: " << numberOfRecTracks << endl << endl;
    
  // Write the histos to file
  theFile->cd();
  hPres->Write();
  h1_Pres->Write();

  charge->Write();
  ptot->Write();
  pt->Write();
  px->Write();
  py->Write();
  pz->Write();
  Nmuon->Write();
  Nrechits->Write();
  NDThits->Write();
  NCSChits->Write();
  DTvsCSC->Write();
  theFile->Close();
}


void PhysicsObjectsMonitor::analyze(const Event & event, const EventSetup& eventSetup){
  
  cout << "Run: " << event.id().run() << " Event: " << event.id().event() << endl;
  MuonPatternRecoDumper debug;
  
  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> staTracks;
  event.getByLabel(theSTAMuonLabel, staTracks);

  ESHandle<MagneticField> theMGField;
  eventSetup.get<IdealMagneticFieldRecord>().get(theMGField);

  
  double recPt=0.;
  double simPt=0.;

  // Get the SimTrack collection from the event
  if(theDataType == "SimData"){
    Handle<SimTrackContainer> simTracks;
    event.getByLabel("g4SimHits",simTracks);
    
    numberOfRecTracks += staTracks->size();

    SimTrackContainer::const_iterator simTrack;

    cout<<"Simulated tracks: "<<endl;
    for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack){
      if (abs((*simTrack).type()) == 13) {
	cout<<"Sim pT: "<<(*simTrack).momentum().perp()<<endl;
	simPt=(*simTrack).momentum().perp();
	cout<<"Sim Eta: "<<(*simTrack).momentum().eta()<<endl;
	numberOfSimTracks++;
      }    
    }
    cout << endl;
  }
  
  reco::TrackCollection::const_iterator staTrack;
  
  cout<<"Reconstructed tracks: " << staTracks->size() << endl;
  Nmuon->Fill(staTracks->size());
  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack){
    reco::TransientTrack track(*staTrack,&*theMGField); 
    
    int nrechits=0;
    int nDThits=0;
    int nCSChits=0;
    for (trackingRecHit_iterator it = track.recHitsBegin ();  it != track.recHitsEnd (); it++) {
      if ((*it)->isValid ()) {	    
	std::cout << "Analyzer:  Aha this looks like a Rechit!" << std::endl;
	if((*it)->geographicalId().subdetId() == MuonSubdetId::DT) {
	  nDThits++;
	} else if((*it)->geographicalId().subdetId() == MuonSubdetId::CSC) {
	  nCSChits++;
	} else {
	  std::cout << "This is an UNKNOWN hit !! " << std::endl;
	}
	nrechits++;
      }
    }
		
    Nrechits->Fill(nrechits);
    NDThits->Fill(nDThits);
    NCSChits->Fill(nCSChits);
    DTvsCSC->Fill(nDThits,nCSChits);

    debug.dumpFTS(track.impactPointTSCP().theState());
    
    recPt = track.impactPointTSCP().momentum().perp();    
    cout<<" p: "<<track.impactPointTSCP().momentum().mag()<< " pT: "<<recPt<<endl;
    pt->Fill(recPt);
    ptot->Fill(track.impactPointTSCP().momentum().mag());
    charge->Fill(track.impactPointTSCP().charge());
    px->Fill(track.impactPointTSCP().momentum().x());
    py->Fill(track.impactPointTSCP().momentum().y());
    pz->Fill(track.impactPointTSCP().momentum().z());
  }
  cout<<"---"<<endl;
  if(recPt && theDataType == "SimData"){
    hPres->Fill( (recPt-simPt)/simPt);
    h1_Pres->Fill( ( 1/recPt - 1/simPt)/ (1/simPt));

  }  
}

DEFINE_FWK_MODULE(PhysicsObjectsMonitor)
