/** \class STAMuonAnalyzer
 *  Analyzer of the StandAlone muon tracks
 *
 *  $Date: 2009/10/31 05:19:45 $
 *  $Revision: 1.7 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 *  \modified by C. Calabria 
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

#include <DataFormats/GEMRecHit/interface/GEMRecHit.h>
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
 
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include <Geometry/GEMGeometry/interface/GEMEtaPartition.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include "DataFormats/Provenance/interface/Timestamp.h"

#include <DataFormats/MuonDetId/interface/GEMDetId.h>

#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"

#include "RecoMuon/DetLayers/interface/MuRodBarrelLayer.h"
#include "RecoMuon/DetLayers/interface/MuDetRod.h"
#include "RecoMuon/DetLayers/interface/MuRingForwardDoubleLayer.h"
#include "RecoMuon/DetLayers/interface/MuDetRing.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include <DataFormats/MuonDetId/interface/DTWireId.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/GEMDetId.h>

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

using namespace std;
using namespace edm;

/// Constructor
STAMuonAnalyzer::STAMuonAnalyzer(const ParameterSet& pset):
  histContainer_(),
  histContainer2D_()
{
  staTrackLabel_ = pset.getUntrackedParameter<edm::InputTag>("StandAloneTrackCollectionLabel");
  theSeedCollectionLabel = pset.getUntrackedParameter<string>("MuonSeedCollectionLabel");

  theRootFileName = pset.getUntrackedParameter<string>("rootFileName");

  theDataType = pset.getUntrackedParameter<string>("DataType");
  
  if(theDataType != "RealData" && theDataType != "SimData")
    cout<<"Error in Data Type!!"<<endl;

  noGEMCase_ = pset.getUntrackedParameter<bool>("NoGEMCase");
  isGlobalMuon_ = pset.getUntrackedParameter<bool>("isGlobalMuon",false);
  includeME11_ = pset.getUntrackedParameter<bool>("includeME11",true);

  numberOfSimTracks=0;
  numberOfRecTracks=0;
}

/// Destructor
STAMuonAnalyzer::~STAMuonAnalyzer(){
}

void STAMuonAnalyzer::beginJob(){
  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theFile->cd();

  hPtRec = new TH1F("pTRec","p_{T}^{rec}",261,-2.5,1302.5);
  hDeltaPtRec = new TH1F("DeltapTRec","#Delta p_{T}^{rec}",400,-2,2);
  hPtSim = new TH1F("pTSim","p_{T}^{gen} ",261,-2.5,1302.5);

  hPTDiff = new TH1F("pTDiff","p_{T}^{rec} - p_{T}^{gen} ",400,-1000,1000);
  hPTDiff2 = new TH1F("pTDiff2","p_{T}^{rec} - p_{T}^{gen} ",400,-1000,1000);

  hPTDiffvsEta = new TH2F("PTDiffvsEta","p_{T}^{rec} - p_{T}^{gen} VS #eta",100,-2.5,2.5,200,-1000,1000);
  hPTDiffvsPhi = new TH2F("PTDiffvsPhi","p_{T}^{rec} - p_{T}^{gen} VS #phi",100,-6,6,200,-1000,1000);

  hPres = new TH1F("pTRes","pT Resolution",400,-2,2);
  h1_Pres = new TH1F("invPTRes","1/pT Resolution",400,-2,2);

  hSimEta = new TH1F("PSimEta","SimTrack #eta",100,-2.5,2.5);
  hRecEta = new TH1F("PRecEta","RecTrack #eta",100,-2.5,2.5);
  hDeltaEta = new TH1F("PDeltaEta","#Delta#eta",100,-1,1);
  hDeltaPhi = new TH1F("PDeltaPhi","#Delta#phi",100,-1,1);
  hDeltaPhiPlus = new TH1F("PDeltaPhiPlus","#Delta#phi q>0",100,-1,1);
  hDeltaPhiMinus = new TH1F("PDeltaPhiMinus","#Delta#phi q<0",100,-1,1);

  hSimPhi = new TH1F("PSimPhi","SimTrack #phi",100,-TMath::Pi(),TMath::Pi());
  hRecPhi = new TH1F("PRecPhi","RecTrack #phi",100,-TMath::Pi(),TMath::Pi());
  hRecPhi2DPlusLayer1 = new TH2F("RecHitPhi2DPlusLayer1","RecHit #phi GE+1 vs. Sector, Layer 1",36,-TMath::Pi(),TMath::Pi(),36,0,36);
  hRecPhi2DMinusLayer1 = new TH2F("RecHitPhi2DMinusLayer1","RecHit #phi GE-1 vs. Sector, Layer 1",36,-TMath::Pi(),TMath::Pi(),36,0,36);
  hRecPhi2DPlusLayer2 = new TH2F("RecHitPhi2DPlusLayer2","RecHit #phi GE+1 vs. Sector, Layer 2",36,-TMath::Pi(),TMath::Pi(),36,0,36);
  hRecPhi2DMinusLayer2 = new TH2F("RecHitPhi2DMinusLayer2","RecHit #phi GE-1 vs. Sector, Layer 2",36,-TMath::Pi(),TMath::Pi(),36,0,36);

  hNumSimTracks = new TH1F("NumSimTracks","NumSimTracks",100,0,100);
  hNumMuonSimTracks = new TH1F("NumMuonSimTracks","NumMuonSimTracks",10,0,10);
  hNumRecTracks = new TH1F("NumRecTracks","NumRecTracks",10,0,10);
  hNumGEMSimHits = new TH1F("NumGEMSimHits","NumGEMSimHits",10,0,10);
  hNumCSCSimHits = new TH1F("NumCSCSimHits","NumCSCSimHits",10,0,10);
  hNumGEMRecHits = new TH1F("NumGEMRecHits","NumGEMRecHits",10,0,10);
  hNumCSCRecHits = new TH1F("NumCSCRecHits","NumCSCRecHits",10,0,10);

  //Double_t nbins[] = {0,10,30,50,100,150,200,300,500,750,1000};
  hRecoPtVsSimPt = new TH2F("RecoPtVsSimPt","p_{T}^{Reco} vs. p_{T}^{Sim}",261,-2.5,1302.5,261,-2.5,1302.5);
  hDeltaPtVsSimPt = new TH2F("DeltaPtVsSimPt","(p_{T}^{Reco} - p_{T}^{Sim}) vs. p_{T}^{Sim}",261,-2.5,1302.5,500,-500,500);

  hPtResVsPt = new TH2F("PtResVsPt","p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,400,-2,2);
  hInvPtResVsPt = new TH2F("InvPtResVsPt","1/p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,400,-2,2);

  hPtResVsPtNoCharge = new TH2F("PtResVsPtNoCharge","p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,400,-2,2);
  hInvPtResVsPtNoCharge = new TH2F("InvPtResVsPtNoCharge","1/p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,400,-2,2);

  hDPhiVsPt = new TH2F("DPhiVsPt","#Delta#phi vs. p_{T}",261,-2.5,1302.5,100,-6,6);

  hPtResVsEta = new TH2F("PtResVsEta","p_{T} Resolution vs. #eta",100,-2.5,2.5,400,-2,2);
  hInvPtResVsEta = new TH2F("InvPtResVsEta","1/p_{T} Resolution vs. #eta",100,-2.5,2.5,400,-2,2);

  hPtResVsEtaNoCharge = new TH2F("PtResVsEtaNoCharge","p_{T} Resolution vs. #eta",100,-2.5,2.5,400,-2,2);
  hInvPtResVsEtaNoCharge = new TH2F("InvPtResVsEtaNoCharge","1/p_{T} Resolution vs. #eta",100,-2.5,2.5,400,-2,2);

  hDenPt = new TH1F("DenPt","DenPt",261,-2.5,1302.5);
  hDenEta = new TH1F("DenEta","DenEta",100,-2.5,2.5);
  hDenPhi = new TH1F("DenPhi","DenPhi",36,-TMath::Pi(),TMath::Pi());
  hDenPhiPlus = new TH1F("DenPhiPlus","DenPhiMinus",360,0,180);
  hDenPhiMinus = new TH1F("DenPhiMinus","DenPhiMinus",360,0,180);
  hDenSimPt = new TH1F("DenSimPt","DenSimPt",261,-2.5,1302.5);
  hDenSimEta = new TH1F("DenSimEta","DenSimEta",100,-2.5,2.5);
  hDenSimPhiPlus = new TH1F("DenSimPhiPlus","DenSimPhiMinus",360,0,180);
  hDenSimPhiMinus = new TH1F("DenSimPhiMinus","DenSimPhiMinus",360,0,180);
  hNumPt = new TH1F("NumPt","NumPt",261,-2.5,1302.5);
  hNumEta = new TH1F("NumEta","NumEta",100,-2.5,2.5);
  hNumPhi = new TH1F("NumPhi","NumPhi",36,-TMath::Pi(),TMath::Pi());
  hNumPhiPlus = new TH1F("NumPhiPlus","NumPhiMinus",360,0,180);
  hNumPhiMinus = new TH1F("NumPhiMinus","NumPhiMinus",360,0,180);
  hNumSimPt = new TH1F("NumSimPt","NumSimPt",261,-2.5,1302.5);
  hNumSimEta = new TH1F("NumSimEta","NumSimEta",100,-2.5,2.5);
  hNumSimPhiPlus = new TH1F("NumSimPhiPlus","NumSimPhiMinus",360,0,180);
  hNumSimPhiMinus = new TH1F("NumSimPhiMinus","NumSimPhiMinus",360,0,180);

  hPullGEMx = new TH1F("PullGEMx", "(x_{mc} - x_{rec}) / #sigma",500,-10.,10.);
  hPullGEMy = new TH1F("PullGEMy", "(y_{mc} - y_{rec}) / #sigma",500,-10.,10.);
  hPullGEMz = new TH1F("PullGEMz", "(z_{mc} - z_{rec}) / #sigma",500,-10.,10.);
  hPullCSC = new TH1F("PullCSC", "(x_{mc} - x_{rec}) / #sigma",500,-10.,10.);

  hGEMRecHitEta = new TH1F("GEMRecHitEta","GEM RecHits #eta",100,-2.5,2.5);
  hGEMRecHitPhi = new TH1F("GEMRecHitPhi","GEM RecHits #phi",360,-TMath::Pi(),TMath::Pi());

  hDR = new TH1F("DR","#Delta R (SIM-RECO)",300,0,1);
  hDR2 = new TH1F("DRGEM","#Delta R (SIM-RECO)",500,0,0.5);
  hDR3 = new TH1F("DRCSC","#Delta R (SIM-RECO)",500,0,0.5);

  hCharge = new TH2F("Charge","q (SIM-RECO)",6,-3,3,6,-3,3);
  hDeltaCharge = new TH2F("DeltaCharge","#Delta q (SIM-RECO)",261,-2.5,1302.5,6,-3,3);

  hDeltaPhiVsSimTrackPhi = new TH2F("DeltaPhiVsSimTrackPhi","DeltaPhiVsSimTrackPhi",360,0,180,2000,-20,+20);
  hDeltaPhiVsSimTrackPhi2 = new TH2F("DeltaPhiVsSimTrackPhi2","DeltaPhiVsSimTrackPhi2",360,0,180,2000,-20,+20);
  //hPTDiffvsEta = new TH2F("PTDiffvsEta","p_{T}^{rec} - p_{T}^{gen} VS #eta",100,-2.5,2.5,200,-1000,1000);

  hTracksWithME11 = new TH1F("TracksWithME11", "TracksWithME11",2,0.,2.);

  hPtSimCorr = new TH1F("pTSimCorr","p_{T}^{Corr} ",200,0,1000);
  hPtResVsPtCorr = new TH2F("PtResVsPtCorr","p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,400,-2,2);
  hInvPtResVsPtCorr = new TH2F("InvPtResVsPtCorr","1/p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,400,-2,2);

  hCSCorGEM = new TH1F("CSCorGEM", "CSCorGEM",4,0.,4.);
  hSimTrackMatch = new TH1F("SimTrackMatch", "SimTrackMatch",2,0.,2.);
  hRecHitMatching = new TH1F("RecHitMatching", "RecHitMatching",2,0.,2.);

}

void STAMuonAnalyzer::endJob(){
  if(theDataType == "SimData"){
    cout << endl << endl << "Number of Sim tracks: " << numberOfSimTracks << endl;
  }

  cout << "Number of Reco tracks: " << numberOfRecTracks << endl << endl;
    
  // Write the histos to file
  theFile->cd();
  hPtRec->Write();
  hDeltaPtRec->Write();
  hPtSim->Write();
  hPres->Write();
  h1_Pres->Write();
  hPTDiff->Write();
  hPTDiff2->Write();
  hPTDiffvsEta->Write();
  hPTDiffvsPhi->Write();
  hSimEta->Write();
  hRecEta->Write();
  hDeltaEta->Write();
  hDeltaPhi->Write();
  hDeltaPhiPlus->Write();
  hDeltaPhiMinus->Write();
  hSimPhi->Write();
  hRecPhi->Write();
  hNumSimTracks->Write();
  hNumMuonSimTracks->Write();
  hNumRecTracks->Write();
  hNumGEMSimHits->Write();
  hNumCSCSimHits->Write();
  hNumGEMRecHits->Write();
  hNumCSCRecHits->Write();
  hPtResVsPt->Write();
  hInvPtResVsPt->Write();
  hPtResVsEta->Write();
  hInvPtResVsEta->Write();
  hPtResVsPtNoCharge->Write();
  hInvPtResVsPtNoCharge->Write();
  hPtResVsEtaNoCharge->Write();
  hInvPtResVsEtaNoCharge->Write();
  hDPhiVsPt->Write();
  hDenPt->Write();
  hDenEta->Write();
  hDenPhi->Write();
  hDenPhiPlus->Write();
  hDenPhiMinus->Write();
  hDenSimPt->Write();
  hDenSimEta->Write();
  hDenSimPhiPlus->Write();
  hDenSimPhiMinus->Write();
  hNumPt->Write();
  hNumEta->Write();
  hNumPhi->Write();
  hNumPhiPlus->Write();
  hNumPhiMinus->Write();
  hNumSimPt->Write();
  hNumSimEta->Write();
  hNumSimPhiPlus->Write();
  hNumSimPhiMinus->Write();
  hPullGEMx->Write();
  hPullGEMy->Write();
  hPullGEMz->Write();
  hPullCSC->Write();
  hGEMRecHitEta->Write();
  hGEMRecHitPhi->Write();
  hDR->Write();
  hDR2->Write();
  hDR3->Write();
  hRecPhi2DPlusLayer1->Write();
  hRecPhi2DMinusLayer1->Write();
  hRecPhi2DPlusLayer2->Write();
  hRecPhi2DMinusLayer2->Write();
  hCharge->Write();
  hDeltaCharge->Write();
  hDeltaPhiVsSimTrackPhi->Write();
  hDeltaPhiVsSimTrackPhi2->Write();
  hRecoPtVsSimPt->Write();
  hDeltaPtVsSimPt->Write();
  hTracksWithME11->Write();

  hPtSimCorr->Write();
  hPtResVsPtCorr->Write();
  hInvPtResVsPtCorr->Write();
  hCSCorGEM->Write();
  hSimTrackMatch->Write();
  hRecHitMatching->Write();

  theFile->Close();
}

GlobalPoint propagatedPositionGEM(SimTrackContainer::const_iterator simTrack, const Event & event, const EventSetup& eventSetup)
{

  edm::Handle<edm::SimVertexContainer> sim_vertices;
  event.getByLabel("g4SimHits", sim_vertices);

  //int vxtIndex = simTrack->vertIndex();
  //int parentId = (*sim_vertices)[vxtIndex].parentIndex();

  static const float AVERAGE_GEM_Z(568.6); // [cm]

  edm::ESHandle<MagneticField> magfield_;
  edm::ESHandle<Propagator> propagator_;
  edm::ESHandle<Propagator> propagatorOpposite_;

  // Get the magnetic field
  eventSetup.get< IdealMagneticFieldRecord >().get(magfield_);

  // Get the propagators
  eventSetup.get< TrackingComponentsRecord >().get("SteppingHelixPropagatorAlong", propagator_);
  eventSetup.get< TrackingComponentsRecord >().get("SteppingHelixPropagatorOpposite", propagatorOpposite_);

  const double eta((*simTrack).momentum().eta());
  const int endcap((eta > 0.) ? 1 : -1);

  GlobalPoint inner_point((*sim_vertices)[simTrack->vertIndex()].position().x(), (*sim_vertices)[simTrack->vertIndex()].position().y(), (*sim_vertices)[simTrack->vertIndex()].position().z());
  GlobalVector inner_vec ((*simTrack).momentum().x(), (*simTrack).momentum().y(), (*simTrack).momentum().z());

  Plane::PositionType pos(0.f, 0.f, endcap*AVERAGE_GEM_Z);
  Plane::RotationType rot;
  Plane::PlanePointer my_plane(Plane::build(pos, rot));

  FreeTrajectoryState state_start(inner_point, inner_vec, (*simTrack).charge(), &*magfield_);

  TrajectoryStateOnSurface tsos(propagator_->propagate(state_start, *my_plane));
  if (!tsos.isValid()) tsos = propagatorOpposite_->propagate(state_start, *my_plane);

  if (tsos.isValid()) return tsos.globalPosition();
  return GlobalPoint();

}

bool isSimMatched(SimTrackContainer::const_iterator simTrack, edm::PSimHitContainer::const_iterator itHit)
{

  bool result = false;

  int trackId = simTrack->trackId();
  int trackId_sim = itHit->trackId();
  if(trackId == trackId_sim) result = true;

  //std::cout<<"ID: "<<trackId<<" "<<trackId_sim<<" "<<result<<std::endl;

  return result;

}

edm::PSimHitContainer isTrackMatched(SimTrackContainer::const_iterator simTrack, const Event & event, const EventSetup& eventSetup)
{

  edm::PSimHitContainer selectedGEMHits;

  //GlobalPoint gbTemp = propagatedPositionGEM(simTrack, event, eventSetup);
  //std::cout<<gbTemp.x()<<std::endl;

  edm::Handle<edm::PSimHitContainer> GEMHits;
  event.getByLabel(edm::InputTag("g4SimHits","MuonGEMHits"), GEMHits);

  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  eventSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

  for (edm::PSimHitContainer::const_iterator itHit = GEMHits->begin(); itHit != GEMHits->end(); ++itHit){
							 
	if(itHit->particleType() != (*simTrack).type()) continue;
	DetId id = DetId(itHit->detUnitId());
	if (!(id.subdetId() == MuonSubdetId::GEM)) continue;
  	if(itHit->particleType() != (*simTrack).type()) continue;

	bool result = isSimMatched(simTrack, itHit);
	if(result) selectedGEMHits.push_back(*itHit);

  }

  //std::cout<<"Size: "<<selectedGEMHits.size()<<std::endl;
  return selectedGEMHits;

}

bool isRecHitMatched(edm::PSimHitContainer selGEMSimHits, trackingRecHit_iterator recHit, edm::ESHandle<GEMGeometry> gemGeom)
{

  bool result = false;

  GEMDetId id((*recHit)->geographicalId());
  LocalPoint lp1 = (*recHit)->localPosition();
  int region = id.region();
  int layer = id.layer();
  int chamber = id.chamber();
  int strip = gemGeom->etaPartition(id)->strip(lp1);
 
  for(edm::PSimHitContainer::const_iterator itHit = selGEMSimHits.begin(); itHit != selGEMSimHits.end(); ++itHit){

      	GEMDetId idGem = GEMDetId(itHit->detUnitId());
      	int region_sim = idGem.region();
      	int layer_sim = idGem.layer();
      	int chamber_sim = idGem.chamber();

      	LocalPoint lp = itHit->entryPoint();
      	int strip_sim = gemGeom->etaPartition(idGem)->strip(lp);

	//std::cout<<"Strip: "<<strip<<" "<<strip_sim<<std::endl;

      	if(region != region_sim) continue;
      	if(layer != layer_sim) continue;
      	if(chamber != chamber_sim) continue;

      	if(abs(strip - strip_sim) < 2) result = true;

  }

  //std::cout<<"RecHit: "<<result<<std::endl;
  return result;

}

void STAMuonAnalyzer::analyze(const Event & event, const EventSetup& eventSetup){
  
  //cout << "Run: " << event.id().run() << " Event: " << event.id().event() << endl;
  MuonPatternRecoDumper debug;
  
  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> staTracks;
  event.getByLabel(staTrackLabel_, staTracks);

  Handle<SimTrackContainer> simTracks;
  event.getByLabel("g4SimHits",simTracks);

  ESHandle<MagneticField> theMGField;
  eventSetup.get<IdealMagneticFieldRecord>().get(theMGField);

  edm::ESHandle<GEMGeometry> gemGeom;
  eventSetup.get<MuonGeometryRecord>().get(gemGeom);

  edm::ESHandle<MuonDetLayerGeometry> geo;
  eventSetup.get<MuonRecoGeometryRecord>().get(geo);

  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  eventSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

  edm::Handle<GEMRecHitCollection> gemRecHits; 
  event.getByLabel("gemRecHits","",gemRecHits);

  edm::Handle<CSCRecHit2DCollection> cscRecHits; 
  event.getByLabel("csc2DRecHits","",cscRecHits);

  edm::Handle<edm::PSimHitContainer> GEMHits;
  event.getByLabel(edm::InputTag("g4SimHits","MuonGEMHits"), GEMHits);

  edm::Handle<edm::PSimHitContainer> CSCHits;
  event.getByLabel(edm::InputTag("g4SimHits","MuonCSCHits"), CSCHits);

  double recPt = 0.;
  double recPtIP = 0.;
  double simPt = 0.;
  double simEta = 0.;
  double simPhi = 0.;

  hNumRecTracks->Fill(staTracks->size());

  int simCount = 0;
  //int numRecoTrack = 0;

  // Get the SimTrack collection from the event
  if(theDataType == "SimData"){

	//numRecoTrack = simTracks->size();
  	hNumSimTracks->Fill(simTracks->size());
    
    	numberOfRecTracks += staTracks->size();

    	SimTrackContainer::const_iterator simTrack;

    	//cout<<"Simulated tracks: "<<endl;
    	for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack){

	      	if (!(abs((*simTrack).type()) == 13)) continue; 
  		if ((*simTrack).noVertex()) continue;
  		if ((*simTrack).noGenpart()) continue;

		//cout<<"Sim pT: "<<(*simTrack).momentum().pt()<<endl;
		simPt=(*simTrack).momentum().pt();
		//cout<<"Sim Eta: "<<(*simTrack).momentum().eta()<<endl;
		simEta = (*simTrack).momentum().eta();

		//cout<<"Sim Phi: "<<(*simTrack).momentum().phi()<<endl;
		simPhi = (*simTrack).momentum().phi();

		numberOfSimTracks++;
		simCount++;
		hSimEta->Fill((*simTrack).momentum().eta());
		hSimPhi->Fill((*simTrack).momentum().phi());	

    	}

	hNumMuonSimTracks->Fill(simCount);
    	//cout << endl;

  }
  
  reco::TrackCollection::const_iterator staTrack;
  
  //cout<<"Reconstructed tracks: " << staTracks->size() << endl;

  SimTrackContainer::const_iterator simTrack;

  for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack){

	      	if (abs((*simTrack).type()) != 13) continue;
  		if ((*simTrack).noVertex()) continue;
  		if ((*simTrack).noGenpart()) continue;

		simEta = (*simTrack).momentum().eta();
		simPhi = (*simTrack).momentum().phi();

		if (abs(simEta) > 2.1 || abs(simEta) < 1.64) continue;

		//std::cout<<"SimEta "<<simEta<<" SimPhi "<<simPhi<<std::endl;

		edm::PSimHitContainer selGEMSimHits = isTrackMatched(simTrack, event, eventSetup);
		int size = selGEMSimHits.size();
		hSimTrackMatch->Fill(size > 0 ? 1 : 0);
		if(size == 0) continue;

	  	for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack){//Inizio del loop sulle STA track

		    	reco::TransientTrack track(*staTrack,&*theMGField,theTrackingGeometry); 
		    	TrajectoryStateOnSurface innerTSOS = track.innermostMeasurementState();
		    
		    	//cout << debug.dumpFTS(track.impactPointTSCP().theState());

  			GlobalVector tsosVect = innerTSOS.globalMomentum();
  			math::XYZVectorD reco(tsosVect.x(), tsosVect.y(), tsosVect.z());

			//double recEta = reco.eta();
			//double recPhi = reco.phi();

			double recEta = staTrack->momentum().eta();
			double recPhi = staTrack->momentum().phi();
			//cout<<"RecEta "<<recEta<<" recPhi "<<recPhi<<std::endl;
			double dR = sqrt(pow((simEta-recEta),2) + pow((simPhi-recPhi),2));
			//cout<<"dR "<<dR<<std::endl;

			if(dR > 0.1) continue;
		    
		    	//recPt = track.impactPointTSCP().momentum().perp();  
			recPt = staTrack->pt();
			recPtIP = track.impactPointTSCP().momentum().perp();
		    	//cout<<" p: "<<track.impactPointTSCP().momentum().mag()<< " pT: "<<recPt<<endl;
		    	//cout<<" chi2: "<<track.chi2()<<endl;
		    
		    	hPtRec->Fill(recPt);
		    	hDeltaPtRec->Fill(recPt - recPtIP);

		    	//cout << "Inner TSOS:"<<endl;
		    	//cout << debug.dumpTSOS(innerTSOS);
		    	//cout<<" p: "<<innerTSOS.globalMomentum().mag()<< " pT: "<<innerTSOS.globalMomentum().perp()<<endl;

		    	/*trackingRecHit_iterator rhbegin = staTrack->recHitsBegin();
		    	trackingRecHit_iterator rhend = staTrack->recHitsEnd();
		    
		    	cout<<"RecHits:"<<endl;
		    	for(trackingRecHit_iterator recHit = rhbegin; recHit != rhend; ++recHit){
		      		const GeomDet* geomDet = theTrackingGeometry->idToDet((*recHit)->geographicalId());
		      		//std::cout<<"detID "<<((*recHit)->geographicalId())<<" "<<geomDet<<std::endl;
		      		double r = geomDet->surface().position().perp();
		      		double z = geomDet->toGlobal((*recHit)->localPosition()).z();
		      		//cout<<"r: "<< r <<" z: "<<z <<endl;
		    	}*/
	    
		    	if(recPt && theDataType == "SimData" && abs(recEta) > 1.64 && abs(recEta) < 2.1){

				float phi_02pi = recPhi < 0 ? recPhi + TMath::Pi() : recPhi;
				float phiDeg = phi_02pi * 180/ TMath::Pi();
				//int phiSec = phiDeg%18;
				float phi_02pi_sim = simPhi < 0 ? simPhi + TMath::Pi() : simPhi;
				float phiDegSim = phi_02pi_sim * 180/ TMath::Pi();
				//int phiSec = phiDeg%18;

		    	 	bool hasGemRecHits = false;
				bool hasRecHitsFromCSCME11 = false;
				int numGEMRecHits = 0;
				int numGEMSimHits = 0;
				int numCSCRecHits = 0;
				int numCSCSimHits = 0;

				for (edm::PSimHitContainer::const_iterator itHit = GEMHits->begin(); itHit != GEMHits->end(); ++itHit){
							 
					if(itHit->particleType() != (*simTrack).type()) continue;
					DetId id = DetId(itHit->detUnitId());
					if (!(id.subdetId() == MuonSubdetId::GEM)) continue;

					GlobalPoint pointSimHit = theTrackingGeometry->idToDetUnit(id)->toGlobal(itHit->localPosition());
					float phi_02pi_simHit = pointSimHit.phi() < 0 ? pointSimHit.phi() + TMath::Pi() : pointSimHit.phi();
					float phiDegSimHit = phi_02pi_simHit * 180/ TMath::Pi();

					if(simPt > 100) hDeltaPhiVsSimTrackPhi2->Fill(phiDegSim, phiDegSim - phiDegSimHit);

				}

				std::vector<bool> collectResults;

		      		for(trackingRecHit_iterator recHit = staTrack->recHitsBegin(); recHit != staTrack->recHitsEnd(); ++recHit){

					if ((*recHit)->geographicalId().det() == DetId::Muon){

					if ((*recHit)->geographicalId().subdetId() == MuonSubdetId::GEM){

						//std::cout<<"GEM id: "<<GEMDetId((*recHit)->geographicalId().rawId())<<std::endl;
						numGEMRecHits++;
						hasGemRecHits = true;

						bool status = isRecHitMatched(selGEMSimHits, recHit, gemGeom);
						collectResults.push_back(status);

						if(!isGlobalMuon_){

						   	GEMDetId id((*recHit)->geographicalId());

						    	int region = id.region();
						    	int layer = id.layer();
						    	int chamber = id.chamber();
						    	//int roll = id.roll();

			      				const GeomDet* geomDet = theTrackingGeometry->idToDet((*recHit)->geographicalId());
			      				//double r = geomDet->surface().position().perp();
			      				double x_reco = (*recHit)->localPosition().x();
			      				double err_x_reco = (*recHit)->localPositionError().xx();

			      				double x = geomDet->toGlobal((*recHit)->localPosition()).x();
			      				double y = geomDet->toGlobal((*recHit)->localPosition()).y();
			      				double z = geomDet->toGlobal((*recHit)->localPosition()).z();
							GlobalPoint pointRecHit = GlobalPoint(x,y,z);
							
							hGEMRecHitEta->Fill(pointRecHit.eta());
							hGEMRecHitPhi->Fill(pointRecHit.phi());
							if(region > 0 && layer == 1) hRecPhi2DPlusLayer1->Fill(pointRecHit.phi(), chamber);
							else if(region > 0 && layer == 2) hRecPhi2DPlusLayer2->Fill(pointRecHit.phi(), chamber);
							else if(region < 0 && layer == 1) hRecPhi2DMinusLayer1->Fill(pointRecHit.phi(), chamber);
							else if(region < 0 && layer == 2) hRecPhi2DMinusLayer2->Fill(pointRecHit.phi(), chamber);
							//std::cout<<"Eta GEMRecHits "<<pointRecHit.eta()<<std::endl;
							//std::cout<<"Phi GEMRecHits "<<pointRecHit.phi()<<std::endl;

			  				for (edm::PSimHitContainer::const_iterator itHit = GEMHits->begin(); itHit != GEMHits->end(); ++itHit){
							 
								if(itHit->particleType() != (*simTrack).type()) continue;

								DetId id = DetId(itHit->detUnitId());
								if (!(id.subdetId() == MuonSubdetId::GEM)) continue;

								GEMDetId idGem = GEMDetId(itHit->detUnitId());
						    		int region_sim = idGem.region();
						    		int layer_sim = idGem.layer();
						    		//int chamber_sim = id.chamber();
						    		//int roll_sim = id.roll();
								if (!(region == region_sim && layer == layer_sim)) continue;

        						      	GlobalPoint pointSimHit = theTrackingGeometry->idToDetUnit(id)->toGlobal(itHit->localPosition());

								float x_sim = itHit->localPosition().x();
								//float y_sim = itHit->localPosition().y();
								//float z_sim = itHit->localPosition().z();
								double dR2 = deltaR(pointRecHit.eta(),pointRecHit.phi(),pointSimHit.eta(),pointSimHit.phi());
								hDR2->Fill(dR2);
								if(dR2 > 0.1) continue;

								float dX = x_sim - x_reco;
								float pullX = dX/std::sqrt(err_x_reco);
								numGEMSimHits++;
								hPullGEMx->Fill(pullX);

								float phi_02pi_simHit = pointSimHit.phi() < 0 ? pointSimHit.phi() + TMath::Pi() : pointSimHit.phi();
								float phiDegSimHit = phi_02pi_simHit * 180/ TMath::Pi();

								if(simPt > 100) hDeltaPhiVsSimTrackPhi->Fill(phiDegSim, phiDegSim - phiDegSimHit);

							}

						}

					}

					else if((*recHit)->geographicalId().subdetId() == MuonSubdetId::CSC){

						CSCDetId id((*recHit)->geographicalId());
						//int endcap = id.endcap();
						int ring = id.ring();
						int station = id.station();
						//int layer = id.layer();
						//int chamber = id.chamber();

						//std::cout<<"CSC id: "<<CSCDetId((*recHit)->geographicalId().rawId())<<std::endl;
						numCSCRecHits++;
						if(station == 1 && ring == 1) hasRecHitsFromCSCME11 = true;

						if(!isGlobalMuon_){

			      				const GeomDet* geomDet = theTrackingGeometry->idToDet((*recHit)->geographicalId());
			      				double x = geomDet->toGlobal((*recHit)->localPosition()).x();
			      				double y = geomDet->toGlobal((*recHit)->localPosition()).y();
			      				double z = geomDet->toGlobal((*recHit)->localPosition()).z();
							GlobalPoint pointRecHit = GlobalPoint(x,y,z);

			      				double x_reco = (*recHit)->localPosition().x();
			      				double err_x_reco = (*recHit)->localPositionError().xx();

			  				for (edm::PSimHitContainer::const_iterator itHit = CSCHits->begin(); itHit != CSCHits->end(); ++itHit){
							 
								if(itHit->particleType() != (*simTrack).type()) continue;

								DetId id = DetId(itHit->detUnitId());
								if (!(id.subdetId() == MuonSubdetId::CSC)) continue;

								CSCDetId idCsc = CSCDetId(itHit->detUnitId());
						    		//int layer_sim = idCsc.layer();
						    		int station_sim = idCsc.station();
						    		//int chamber_sim = idCsc.chamber();
						    		//int roll_sim = idCsc.roll();
								if (!(station == station_sim)) continue;

        						      	GlobalPoint pointSimHit = theTrackingGeometry->idToDetUnit(id)->toGlobal(itHit->localPosition());

								float x_sim = itHit->localPosition().x();
								double dR2 = deltaR(pointRecHit.eta(),pointRecHit.phi(),pointSimHit.eta(),pointSimHit.phi());
								hDR3->Fill(dR2);
								if(dR2 > 0.1) continue;

								//float err_x_sim = itHit->localPositionError().xx();
								//float y_sim = itHit->localPosition().y();
								float dX = x_sim - x_reco;
								float pull = dX/std::sqrt(err_x_reco);
								numCSCSimHits++;
								hPullCSC->Fill(pull);

							}

						}

					}

					}

		      		}

  				hNumGEMRecHits->Fill(numGEMRecHits);
  				hNumGEMSimHits->Fill(numGEMSimHits);
  				hNumCSCRecHits->Fill(numCSCRecHits);
  				hNumCSCSimHits->Fill(numCSCSimHits);

				//std::cout<<"CSC ME1/1: "<<hasRecHitsFromCSCME11<<std::endl;
				hTracksWithME11->Fill(hasRecHitsFromCSCME11);

				if(hasGemRecHits == true && hasRecHitsFromCSCME11 == true) hCSCorGEM->Fill(0.5);
				else if(hasGemRecHits == true && hasRecHitsFromCSCME11 == false) hCSCorGEM->Fill(1.5);
				else if(hasGemRecHits == false && hasRecHitsFromCSCME11 == true) hCSCorGEM->Fill(2.5);
				else if(hasGemRecHits == false && hasRecHitsFromCSCME11 == false) hCSCorGEM->Fill(3.5);

				if(includeME11_) hasRecHitsFromCSCME11 = true;

				bool matchingHit = true;
				for(int i = 0; i < (int)collectResults.size(); i++){

					matchingHit &= collectResults[i];

				}
				hRecHitMatching->Fill(matchingHit);
				//std::cout<<"Result "<<matchingHit<<std::endl;

				double simPtCorr = 0;
		      		if(noGEMCase_){ 

					hasGemRecHits = true;
					matchingHit = true;
					simPtCorr = (recPt - 0.00115)/0.9998;

				}
				else simPtCorr = (recPt - 0.0005451)/0.9999;

		      		if(hasGemRecHits & matchingHit & (includeME11_ ? hasRecHitsFromCSCME11 : !hasRecHitsFromCSCME11)){

					//TH1::StatOverflows(kTRUE);

					int qGen = simTrack->charge();
					int qRec = staTrack->charge();

					hCharge->Fill(qGen,qRec);
					hDeltaCharge->Fill(simPt,qGen-qRec);

					//cout<<"RecEta "<<recEta<<" recPhi "<<recPhi<<std::endl;
					//cout<<"SimEta "<<simEta<<" SimPhi "<<simPhi<<std::endl;
					//cout<<"dR "<<dR<<std::endl;

					hDR->Fill(dR);
					hRecoPtVsSimPt->Fill(simPt, recPt);
					hDeltaPtVsSimPt->Fill(simPt, recPt - simPt);

		      			hPres->Fill((recPt-simPt)/simPt);

		      			hPtResVsPt->Fill(simPt, (recPt*qRec-simPt*qGen)/(simPt*qGen));
		      			hPtResVsEta->Fill(simEta, (recPt*qRec-simPt*qGen)/(simPt*qGen));

		      			hPtResVsPtNoCharge->Fill(simPt, (recPt-simPt)/simPt);
		      			hPtResVsEtaNoCharge->Fill(simEta, (recPt-simPt)/simPt);

		      			hPtSim->Fill(simPt);

		      			hPtResVsPtCorr->Fill(simPt, (simPtCorr-simPt)/simPt);
		      			hInvPtResVsPtCorr->Fill(simPt, (qRec/simPtCorr - qGen/simPt)/(qGen/simPt));
		      			hPtSimCorr->Fill(simPtCorr);

		      			hPTDiff->Fill(recPt-simPt);
		      			hRecEta->Fill(recEta);

		      			hDeltaEta->Fill(simEta - recEta);
		      			hDeltaPhi->Fill(simPhi - recPhi);
					if(track.charge() > 0) hDeltaPhiPlus->Fill(simPhi - recPhi);
					else if(track.charge() < 0) hDeltaPhiMinus->Fill(simPhi - recPhi);

		      			hRecPhi->Fill(recPhi);
		      			//hPTDiff2->Fill(track.innermostMeasurementState().globalMomentum().perp()-simPt);
		      			hPTDiffvsEta->Fill(recEta,recPt-simPt);
		      			hPTDiffvsPhi->Fill(recPhi,recPt-simPt);

		      			//if( ((recPt-simPt)/simPt) <= -0.4)
					//	cout<<"Out of Res: "<<(recPt-simPt)/simPt<<endl;

		      			h1_Pres->Fill((qRec/recPt - qGen/simPt)/(qGen/simPt));

		      			hInvPtResVsPtNoCharge->Fill(simPt, (1/recPt - 1/simPt)/(1/simPt));
		      			hInvPtResVsEtaNoCharge->Fill(simEta, (1/recPt - 1/simPt)/(1/simPt));

		      			hInvPtResVsPt->Fill(simPt, (qRec/recPt - qGen/simPt)/(qGen/simPt));
		      			hInvPtResVsEta->Fill(simEta, (qRec/recPt - qGen/simPt)/(qGen/simPt));

		      			hDPhiVsPt->Fill(simPt, recPhi-simPhi);

		      			hNumPt->Fill(recPt);
		      			hNumEta->Fill(recEta);
		      			hNumSimPt->Fill(simPt);
		      			hNumSimEta->Fill(simEta);
		      			hNumPhi->Fill(phi_02pi);

		      			if(recEta > 0) hNumPhiPlus->Fill(phiDeg);
		      			else if(recEta < 0) hNumPhiMinus->Fill(phiDeg);
		      			if(simEta > 0) hNumSimPhiPlus->Fill(phiDegSim);
		      			else if(simEta < 0) hNumSimPhiMinus->Fill(phiDegSim);

		      		}

				if(includeME11_ ? hasRecHitsFromCSCME11 : !hasRecHitsFromCSCME11){

			      		hDenPt->Fill(recPt);
			      		hDenEta->Fill(recEta);
			      		hDenSimPt->Fill(simPt);
			      		hDenSimEta->Fill(simEta);

			      		hDenPhi->Fill(phi_02pi);
			      		if(recEta > 0) hDenPhiPlus->Fill(phiDeg);
			      		else if(recEta < 0) hDenPhiMinus->Fill(phiDeg);
			      		if(simEta > 0) hDenSimPhiPlus->Fill(phiDegSim);
			      		else if(simEta < 0) hDenSimPhiMinus->Fill(phiDegSim);

				}

		    	}
    
  	}//Fine loop sulle STA track

  }//Fine loop sulle SimTrack


  //cout<<"---"<<endl;  
}

DEFINE_FWK_MODULE(STAMuonAnalyzer);
