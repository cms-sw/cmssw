/** \class STAMuonAnalyzer
 *  Analyzer of the StandAlone muon tracks
 *
 *  $Date: 2006/08/02 08:08:30 $
 *  $Revision: 1.2 $
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

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "TFile.h"
#include "TH1F.h"

using namespace std;
using namespace edm;

/// Constructor
STAMuonAnalyzer::STAMuonAnalyzer(const ParameterSet& pset){
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
STAMuonAnalyzer::~STAMuonAnalyzer(){
}

void STAMuonAnalyzer::beginJob(const EventSetup& eventSetup){
  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theFile->cd();

  hPres = new TH1F("pTRes","pT Resolution",100,-2,2);
  h1_Pres = new TH1F("invPTRes","1/pT Resolution",100,-2,2);
}

void STAMuonAnalyzer::endJob(){
  if(theDataType == "SimData"){
    cout << endl << endl << "Number of Sim tracks: " << numberOfSimTracks << endl;
  }

  cout << "Number of Reco tracks: " << numberOfRecTracks << endl << endl;
    
  // Write the histos to file
  theFile->cd();
  hPres->Write();
  h1_Pres->Write();
  theFile->Close();
}


void STAMuonAnalyzer::analyze(const Event & event, const EventSetup& eventSetup){
  
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
  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack){
    reco::TransientTrack track(*staTrack,&*theMGField); 
    
    cout << debug.dumpFTS(track.impactPointTSCP().theState());
    
    recPt = track.impactPointTSCP().momentum().perp();    
    cout<<" p: "<<track.impactPointTSCP().momentum().mag()<< " pT: "<<recPt<<endl;
  }
  cout<<"---"<<endl;
  if(recPt && theDataType == "SimData"){
    hPres->Fill( (recPt-simPt)/simPt);
    h1_Pres->Fill( ( 1/recPt - 1/simPt)/ (1/simPt));

  }  
}

DEFINE_FWK_MODULE(STAMuonAnalyzer)
