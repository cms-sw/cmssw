/** \class GLBMuonAnalyzer
 *  Analyzer of the Global muon tracks
 *
 *  $Date: 2010/02/11 00:14:18 $
 *  $Revision: 1.5 $
 *  \author R. Bellan  - INFN Torino       <riccardo.bellan@cern.ch>
 *  \author A. Everett - Purdue University <adam.everett@cern.ch>
 */

#include "RecoMuon/GlobalMuonProducer/test/GLBMuonAnalyzer.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
GLBMuonAnalyzer::GLBMuonAnalyzer(const ParameterSet& pset){
  theGLBMuonLabel = pset.getUntrackedParameter<edm::InputTag>("GlobalTrackCollectionLabel");

  theRootFileName = pset.getUntrackedParameter<string>("rootFileName");

  theDataType = pset.getUntrackedParameter<string>("DataType");
  
  if(theDataType != "RealData" && theDataType != "SimData")
    LogTrace("Analyzer")<<"Error in Data Type!!"<<endl;

  numberOfSimTracks=0;
  numberOfRecTracks=0;
}

/// Destructor
GLBMuonAnalyzer::~GLBMuonAnalyzer(){
}

void GLBMuonAnalyzer::beginJob(){
  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theFile->cd();

  hPtRec = new TH1F("pTRec","p_{T}^{rec}",250,0,120);
  hPtSim = new TH1F("pTSim","p_{T}^{gen} ",250,0,120);

  hPTDiff = new TH1F("pTDiff","p_{T}^{rec} - p_{T}^{gen} ",250,-120,120);
  hPTDiff2 = new TH1F("pTDiff2","p_{T}^{rec} - p_{T}^{gen} ",250,-120,120);

  hPTDiffvsEta = new TH2F("PTDiffvsEta","p_{T}^{rec} - p_{T}^{gen} VS #eta",100,-2.5,2.5,250,-120,120);
  hPTDiffvsPhi = new TH2F("PTDiffvsPhi","p_{T}^{rec} - p_{T}^{gen} VS #phi",100,-6,6,250,-120,120);

  hPres = new TH1F("pTRes","pT Resolution",100,-2,2);
  h1_Pres = new TH1F("invPTRes","1/pT Resolution",100,-2,2);
}

void GLBMuonAnalyzer::endJob(){
  if(theDataType == "SimData"){
    LogTrace("Analyzer") << endl << endl << "Number of Sim tracks: " << numberOfSimTracks << endl;
  }

  LogTrace("Analyzer") << "Number of Reco tracks: " << numberOfRecTracks << endl << endl;
    
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
 

void GLBMuonAnalyzer::analyze(const Event & event, const EventSetup& eventSetup){
  
  //LogTrace("Analyzer") << "Run: " << event.id().run() << " Event: " << event.id().event() << endl;
  MuonPatternRecoDumper debug;
  
  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> staTracks;
  event.getByLabel(theGLBMuonLabel, staTracks);

  ESHandle<MagneticField> theMGField;
  eventSetup.get<IdealMagneticFieldRecord>().get(theMGField);

  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  eventSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);
  
  double recPt=0.;
  double simPt=0.;

  // Get the SimTrack collection from the event
  if(theDataType == "SimData"){
    Handle<SimTrackContainer> simTracks;
    event.getByLabel("g4SimHits",simTracks);
    
    numberOfRecTracks += staTracks->size();

    SimTrackContainer::const_iterator simTrack;
    
    //LogTrace("Analyzer")<<"Simulated tracks: "<<endl;
    for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack){
      if (abs((*simTrack).type()) == 13) {
	//LogTrace("Analyzer")<<"Sim pT: "<<(*simTrack).momentum().perp()<<endl;
	simPt=(*simTrack).momentum().pt();
	//LogTrace("Analyzer")<<"Sim Eta: "<<(*simTrack).momentum().eta()<<endl;
	numberOfSimTracks++;
      }    
    }
    LogTrace("Analyzer") << endl;
  }
  
  reco::TrackCollection::const_iterator staTrack;
  
  //LogTrace("Analyzer")<<"Reconstructed tracks: " << staTracks->size() << endl;

  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack){
    reco::TransientTrack track(*staTrack,&*theMGField,theTrackingGeometry); 
    
    //LogTrace("Analyzer") << debug.dumpFTS(track.impactPointTSCP().theState());
    
    recPt = track.impactPointTSCP().momentum().perp();    
    //LogTrace("Analyzer")<<" p: "<<track.impactPointTSCP().momentum().mag()<< " pT: "<<recPt<<endl;
    //LogTrace("Analyzer")<<" chi2: "<<track.chi2()<<endl;
    
    hPtRec->Fill(recPt);
    
    TrajectoryStateOnSurface innerTSOS = track.innermostMeasurementState();
    //LogTrace("Analyzer") << "Inner TSOS:"<<endl;
    //LogTrace("Analyzer") << debug.dumpTSOS(innerTSOS);
    //LogTrace("Analyzer")<<" p: "<<innerTSOS.globalMomentum().mag()<< " pT: "<<innerTSOS.globalMomentum().perp()<<endl;

    trackingRecHit_iterator rhbegin = staTrack->recHitsBegin();
    trackingRecHit_iterator rhend = staTrack->recHitsEnd();
    
    int muHit=0;
    int tkHit=0;
    
    //LogTrace("Analyzer")<<"RecHits:"<<endl;
    for(trackingRecHit_iterator recHit = rhbegin; recHit != rhend; ++recHit){
      //      const GeomDet* geomDet = theTrackingGeometry->idToDet((*recHit)->geographicalId());
//      double r = geomDet->surface().position().perp();
//      double z = geomDet->toGlobal((*recHit)->localPosition()).z();
      //LogTrace("Analyzer")<<"r: "<< r <<" z: "<<z <<endl;
      if((*recHit)->geographicalId().det() == DetId::Muon) ++muHit;
      if((*recHit)->geographicalId().det() == DetId::Tracker) ++tkHit;
    }
    if(tkHit == 0) LogTrace("GlobalMuonAnalyzer") << "+++++++++ This track does not contain TH hits +++++++++ ";


    
    if(recPt && theDataType == "SimData"){  

      hPres->Fill( (recPt-simPt)/simPt);
      hPtSim->Fill(simPt);

      hPTDiff->Fill(recPt-simPt);

      //      hPTDiff2->Fill(track.innermostMeasurementState().globalMomentum().perp()-simPt);
      hPTDiffvsEta->Fill(track.impactPointTSCP().position().eta(),recPt-simPt);
      hPTDiffvsPhi->Fill(track.impactPointTSCP().position().phi(),recPt-simPt);

      if( ((recPt-simPt)/simPt) <= -0.4)
	LogTrace("Analyzer")<<"Out of Res: "<<(recPt-simPt)/simPt<<endl;
      h1_Pres->Fill( ( 1/recPt - 1/simPt)/ (1/simPt));
    }

    
  }
  LogTrace("Analyzer")<<"---"<<endl;  
}

