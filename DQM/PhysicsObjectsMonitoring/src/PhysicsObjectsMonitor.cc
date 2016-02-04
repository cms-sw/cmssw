/** \class PhysicsObjectsMonitor
 *  Analyzer of the StandAlone muon tracks
 *
 *  $Date: 2009/12/14 22:22:12 $
 *  $Revision: 1.9 $
 *  \author M. Mulders - CERN <martijn.mulders@cern.ch>
 *  Based on STAMuonAnalyzer by R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "DQM/PhysicsObjectsMonitoring/interface/PhysicsObjectsMonitor.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

// Not needed for data monitoring! #include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


using namespace std;
using namespace edm;

/// Constructor
PhysicsObjectsMonitor::PhysicsObjectsMonitor(const ParameterSet& pset){
  theSTAMuonLabel = pset.getUntrackedParameter<string>("StandAloneTrackCollectionLabel");
  theSeedCollectionLabel = pset.getUntrackedParameter<string>("MuonSeedCollectionLabel");

  theRootFileName = pset.getUntrackedParameter<string>("rootFileName");
  saveRootFile = pset.getUntrackedParameter<bool>("produceRootFile");

  theDataType = pset.getUntrackedParameter<string>("DataType");
  
  if(theDataType != "RealData" && theDataType != "SimData")
  edm::LogInfo ("PhysicsObjectsMonitor") <<  "Error in Data Type!!"<<endl;

  numberOfSimTracks=0;
  numberOfRecTracks=0;

  /// get hold of back-end interface
  dbe = edm::Service<DQMStore>().operator->();
  



}

/// Destructor
PhysicsObjectsMonitor::~PhysicsObjectsMonitor(){
}

void PhysicsObjectsMonitor::beginJob(){

  dbe->setCurrentFolder("PhysicsObjects/MuonReconstruction");           


  hPres = dbe->book1D("pTRes","pT Resolution",100,-2,2);
  h1_Pres =dbe->book1D("invPTRes","1/pT Resolution",100,-2,2);

  charge= dbe->book1D("charge","track charge",5,-2.5,2.5);
  ptot = dbe->book1D("ptot","track momentum",50,0,50);
  pt = dbe->book1D("pt","track pT",100,0,50);
  px = dbe->book1D("px ","track px",100,-50,50);
  py = dbe->book1D("py","track py",100,-50,50);
  pz = dbe->book1D("pz","track pz",100,-50,50);
  Nmuon = dbe->book1D("Nmuon","Number of muon tracks",11,-.5,10.5);
  Nrechits = dbe->book1D("Nrechits","Number of RecHits/Segments on track",21,-.5,21.5);
  NDThits = dbe->book1D("NDThits","Number of DT Hits/Segments on track",31,-.5,31.5);
  NCSChits = dbe->book1D("NCSChits","Number of CSC Hits/Segments on track",31,-.5,31.5);
  NRPChits = dbe->book1D("NRPChits","Number of RPC hits on track",11,-.5,11.5);

  DTvsCSC = dbe->book2D("DTvsCSC","Number of DT vs CSC hits on track",29,-.5,28.5,29,-.5,28.5);
  TH2F * root_ob = DTvsCSC->getTH2F();
  root_ob->SetXTitle("Number of DT hits");
  root_ob->SetYTitle("Number of CSC hits");

}

void PhysicsObjectsMonitor::endJob(){
  
 if(theDataType == "SimData"){
   //     edm::LogInfo ("PhysicsObjectsMonitor") << "Number of Sim tracks: " << numberOfSimTracks;
    edm::LogInfo ("PhysicsObjectsMonitor") << "Sorry! Running this package on simulation is no longer supported! ";
  }

  edm::LogInfo ("PhysicsObjectsMonitor") << "Number of Reco tracks: " << numberOfRecTracks ;

  if(saveRootFile) dbe->save(theRootFileName); 
  dbe->setCurrentFolder("PhysicsObjects/MuonReconstruction");
  dbe->removeContents();


}


void PhysicsObjectsMonitor::analyze(const Event & event, const EventSetup& eventSetup){
  
  edm::LogInfo ("PhysicsObjectsMonitor") << "Run: " << event.id().run() << " Event: " << event.id().event();
  MuonPatternRecoDumper debug;
  
  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> staTracks;
  event.getByLabel(theSTAMuonLabel, staTracks);

  ESHandle<MagneticField> theMGField;
  eventSetup.get<IdealMagneticFieldRecord>().get(theMGField);

  
  double recPt=0.;
  double simPt=0.;

  // Get the SimTrack collection from the event
  //  if(theDataType == "SimData"){
  //  Handle<SimTrackContainer> simTracks;
  //  event.getByLabel("g4SimHits",simTracks);
    
  //  numberOfRecTracks += staTracks->size();

  //    SimTrackContainer::const_iterator simTrack;

  //    edm::LogInfo ("PhysicsObjectsMonitor") <<"Simulated tracks: ";
  //  for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack){
  //    if (abs((*simTrack).type()) == 13) {
  //      edm::LogInfo ("PhysicsObjectsMonitor") <<	"Sim pT: "<<(*simTrack).momentum().perp()<<endl;
  //	simPt=(*simTrack).momentum().perp();
  //	edm::LogInfo ("PhysicsObjectsMonitor") <<"Sim Eta: "<<(*simTrack).momentum().eta()<<endl;
  //	numberOfSimTracks++;
  //   }    
  //  }
  //  edm::LogInfo ("PhysicsObjectsMonitor") << "\n";
  //}
  
  reco::TrackCollection::const_iterator staTrack;
  
  edm::LogInfo ("PhysicsObjectsMonitor") << "Reconstructed tracks: " << staTracks->size() << endl;
  Nmuon->Fill(staTracks->size());
  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack){
    reco::TransientTrack track(*staTrack,&*theMGField); 
    
    int nrechits=0;
    int nDThits=0;
    int nCSChits=0;
    int nRPChits=0;

    for (trackingRecHit_iterator it = track.recHitsBegin ();  it != track.recHitsEnd (); it++) {
      if ((*it)->isValid ()) {	    
	edm::LogInfo ("PhysicsObjectsMonitor") << "Analyzer:  Aha this looks like a Rechit!" << std::endl;
	if((*it)->geographicalId().subdetId() == MuonSubdetId::DT) {
	  nDThits++;
	} else if((*it)->geographicalId().subdetId() == MuonSubdetId::CSC) {
	  nCSChits++;
        } else if((*it)->geographicalId().subdetId() == MuonSubdetId::RPC) {
          nRPChits++;
	} else {
	 edm::LogInfo ("PhysicsObjectsMonitor") <<  "This is an UNKNOWN hit !! " << std::endl;
	}
	nrechits++;
      }
    }
		
    Nrechits->Fill(nrechits);
    NDThits->Fill(nDThits);
    NCSChits->Fill(nCSChits);
    DTvsCSC->Fill(nDThits,nCSChits);
    NRPChits->Fill(nRPChits);

    debug.dumpFTS(track.impactPointTSCP().theState());
    
    recPt = track.impactPointTSCP().momentum().perp();    
    edm::LogInfo ("PhysicsObjectsMonitor") << " p: "<<track.impactPointTSCP().momentum().mag()<< " pT: "<<recPt<<endl;
    pt->Fill(recPt);
    ptot->Fill(track.impactPointTSCP().momentum().mag());
    charge->Fill(track.impactPointTSCP().charge());
    px->Fill(track.impactPointTSCP().momentum().x());
    py->Fill(track.impactPointTSCP().momentum().y());
    pz->Fill(track.impactPointTSCP().momentum().z());
  }
  edm::LogInfo ("PhysicsObjectsMonitor") <<"---"<<endl;
  if(recPt && theDataType == "SimData"){
    hPres->Fill( (recPt-simPt)/simPt);
    h1_Pres->Fill( ( 1/recPt - 1/simPt)/ (1/simPt));

  }  
}

DEFINE_FWK_MODULE(PhysicsObjectsMonitor);
