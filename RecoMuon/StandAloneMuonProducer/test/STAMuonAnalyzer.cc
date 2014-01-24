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

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

using namespace std;
using namespace edm;
using namespace reco;

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
  int bins = 1200;
  int min = -6;
  int max = +6;

  if(isGlobalMuon_){

	bins = 400;
	min = -2;
	max = +2;

  }

  hPtRec = new TH1F("pTRec","p_{T}^{rec}",261,-2.5,1302.5);
  hDeltaPtRec = new TH1F("DeltapTRec","#Delta p_{T}^{rec}",400,-2,2);
  hPtSim = new TH1F("pTSim","p_{T}^{gen} ",261,-2.5,1302.5);

  hPTDiff = new TH1F("pTDiff","p_{T}^{rec} - p_{T}^{gen} ",400,-1000,1000);
  hPTDiff2 = new TH1F("pTDiff2","p_{T}^{rec} - p_{T}^{gen} ",400,-1000,1000);

  hPTDiffvsEta = new TH2F("PTDiffvsEta","p_{T}^{rec} - p_{T}^{gen} VS #eta",100,-2.5,2.5,200,-1000,1000);
  hPTDiffvsPhi = new TH2F("PTDiffvsPhi","p_{T}^{rec} - p_{T}^{gen} VS #phi",100,-6,6,200,-1000,1000);

  hPres = new TH1F("pTRes","pT Resolution",bins,min,max);
  h1_Pres = new TH1F("invPTRes","1/pT Resolution",bins,min,max);
  h1_PresMuon = new TH1F("invPTResMuon","1/pT Resolution",1200,-6,6);
  h1_PresMuon2 = new TH1F("invPTResMuon2","1/pT Resolution",1200,-6,6);

  hSimEta = new TH1F("PSimEta","SimTrack #eta",100,-2.5,2.5);
  hRecEta = new TH1F("PRecEta","RecTrack #eta",100,-2.5,2.5);
  hDeltaEta = new TH1F("PDeltaEta","#Delta#eta",100,-1,1);
  hDeltaPhi = new TH1F("PDeltaPhi","#Delta#phi",400,-2,2);
  hDeltaPhiMuon = new TH1F("PDeltaPhiMuon","#Delta#phi",400,-2,2);
  hDeltaPhiPlus = new TH1F("PDeltaPhiPlus","#Delta#phi q>0",400,-2,2);
  hDeltaPhiMinus = new TH1F("PDeltaPhiMinus","#Delta#phi q<0",400,-2,2);

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
  hNumGEMRecHitsMuon = new TH1F("NumGEMRecHitsMuon","NumGEMRecHitsMuon",10,0,10);
  hNumCSCRecHits = new TH1F("NumCSCRecHits","NumCSCRecHits",10,0,10);

  //Double_t nbins[] = {0,10,30,50,100,150,200,300,500,750,1000};
  hRecoPtVsSimPt = new TH2F("RecoPtVsSimPt","p_{T}^{Reco} vs. p_{T}^{Sim}",261,-2.5,1302.5,261,-2.5,1302.5);
  hDeltaPtVsSimPt = new TH2F("DeltaPtVsSimPt","(p_{T}^{Reco} - p_{T}^{Sim}) vs. p_{T}^{Sim}",261,-2.5,1302.5,500,-500,500);

  hPtResVsPt = new TH2F("PtResVsPt","p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,bins,min,max);
  hInvPtResVsPt = new TH2F("InvPtResVsPt","1/p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,bins,min,max);
  hInvPtResVsPtMuon = new TH2F("InvPtResVsPtMuon","1/p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,bins,min,max);

  hPtResVsPtNoCharge = new TH2F("PtResVsPtNoCharge","p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,bins,min,max);
  hInvPtResVsPtNoCharge = new TH2F("InvPtResVsPtNoCharge","1/p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,bins,min,max);

  hDPhiVsPt = new TH2F("DPhiVsPt","#Delta#phi vs. p_{T}",261,-2.5,1302.5,100,-6,6);

  hPtResVsEta = new TH2F("PtResVsEta","p_{T} Resolution vs. #eta",100,-2.5,2.5,bins,min,max);
  hInvPtResVsEta = new TH2F("InvPtResVsEta","1/p_{T} Resolution vs. #eta",100,-2.5,2.5,bins,min,max);
  hInvPtResVsEtaMuon = new TH2F("InvPtResVsEtaMuon","1/p_{T} Resolution vs. #eta",100,-2.5,2.5,bins,min,max);

  hPtResVsEtaNoCharge = new TH2F("PtResVsEtaNoCharge","p_{T} Resolution vs. #eta",100,-2.5,2.5,bins,min,max);
  hInvPtResVsEtaNoCharge = new TH2F("InvPtResVsEtaNoCharge","1/p_{T} Resolution vs. #eta",100,-2.5,2.5,bins,min,max);

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

  hGEMRecHitEta = new TH1F("GEMRecHitEta","GEM RecHits #eta",10000,-2.5,2.5);
  hGEMRecHitPhi = new TH1F("GEMRecHitPhi","GEM RecHits #phi",360,-TMath::Pi(),TMath::Pi());

  hDR = new TH1F("DR","#Delta R (SIM-RECO)",300,0,1);
  hDR2 = new TH1F("DRGEM","#Delta R (SIM-RECO)",500,0,0.5);
  hDR3 = new TH1F("DRCSC","#Delta R (SIM-RECO)",500,0,0.5);

  hCharge = new TH2F("Charge","q (SIM-RECO)",6,-3,3,6,-3,3);
  hDeltaCharge = new TH2F("DeltaCharge","#Delta q (SIM-RECO)",261,-2.5,1302.5,6,-3,3);
  hDeltaChargeMuon = new TH2F("DeltaChargeMuon","#Delta q (SIM-RECO)",261,-2.5,1302.5,6,-3,3);
  hDeltaChargeVsEta = new TH2F("DeltaChargeVsEta","#Delta q (SIM-RECO) vs. #eta",50,0,2.5,6,-3,3);
  hDeltaChargeVsEtaMuon = new TH2F("DeltaChargeVsEtaMuon","#Delta q (SIM-RECO) vs. #eta",50,0,2.5,6,-3,3);

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
  hRecHitParMatching = new TH1F("RecHitParMatching", "RecHitParMatching",2,0.,2.);
  hDRMatchVsPt = new TH2F("DRMatchVsPt","DRMatchVsPt",261,-2.5,1302.5,10,0,10);
  hDRMatchVsPtMuon = new TH2F("DRMatchVsPtMuon","DRMatchVsPtMupn",261,-2.5,1302.5,10,0,10);

  hMatchedSimHits = new TH1F("MatchedSimHits","MatchedSimHits",6,-0.5,5.5);
  hRecoTracksWithMatchedRecHits = new TH2F("RecoTracksWithMatchedRecHits","RecoTracksWithMatchedRecHits",6,-0.5,5.5,6,-0.5,5.5);
  hDeltaQvsDeltaPt = new TH2F("DeltaQvsDeltaPt","DeltaQvsDeltaPt",100,-2,2,7,-3.5,3.5);
  hCheckGlobalTracksVsPt = new TH2F("CheckGlobalTracksVsPt","CheckGlobalTracksVsPt",261,-2.5,1302.5,4,0,4);
  hCheckTracksVsPt = new TH2F("CheckTracksVsPt","CheckTracksVsPt",261,-2.5,1302.5,8,0,8);

  hPtResVsPtRes = new TH2F("PtResVsPtRes","PtResVsPtRes",400,-2,2,400,-2,2);
  hDeltaPtRes = new TH1F("DeltaPtRes","DeltaPtRes",400,-2,2);

  hCountPresence = new TH1F("CountPresence","CountPresence",2,0,2);

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
  h1_PresMuon->Write();
  h1_PresMuon2->Write();
  hPTDiff->Write();
  hPTDiff2->Write();
  hPTDiffvsEta->Write();
  hPTDiffvsPhi->Write();
  hSimEta->Write();
  hRecEta->Write();
  hDeltaEta->Write();
  hDeltaPhi->Write();
  hDeltaPhiMuon->Write();
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
  hNumGEMRecHitsMuon->Write();
  hNumCSCRecHits->Write();
  hPtResVsPt->Write();
  hInvPtResVsPt->Write();
  hInvPtResVsPtMuon->Write();
  hPtResVsEta->Write();
  hInvPtResVsEta->Write();
  hInvPtResVsEtaMuon->Write();
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
  hDeltaChargeMuon->Write();
  hDeltaChargeVsEta->Write();
  hDeltaChargeVsEtaMuon->Write();
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
  hRecHitParMatching->Write();
  hDRMatchVsPt->Write();
  hDRMatchVsPtMuon->Write();
  hMatchedSimHits->Write();
  hRecoTracksWithMatchedRecHits->Write();
  hDeltaQvsDeltaPt->Write();
  hCheckGlobalTracksVsPt->Write();
  hCheckTracksVsPt->Write();
  hPtResVsPtRes->Write();
  hDeltaPtRes->Write();
  hCountPresence->Write();

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
				 
	DetId id = DetId(itHit->detUnitId());
	if (!(id.subdetId() == MuonSubdetId::GEM)) continue;
  	if(itHit->particleType() != (*simTrack).type()) continue;

	bool result = isSimMatched(simTrack, itHit);
	if(result) selectedGEMHits.push_back(*itHit);

  }

  //std::cout<<"Size: "<<selectedGEMHits.size()<<std::endl;
  return selectedGEMHits;

}

bool isRecHitMatched(edm::PSimHitContainer selGEMSimHits, TrackingRecHitRef recHit, edm::ESHandle<GEMGeometry> gemGeom)
{

  bool result = false;

  GEMDetId id(recHit->geographicalId());
  //LocalPoint lp1 = (*recHit)->localPosition();
  //float strip = gemGeom->etaPartition(id)->strip(lp1);
  int region = id.region();
  int layer = id.layer();
  int chamber = id.chamber();
  int roll = id.roll();
  
  const GEMRecHit * gemrechit = dynamic_cast<const GEMRecHit *>(&(*recHit));
  int cls = gemrechit->clusterSize();
  int firstStrip = gemrechit->firstClusterStrip();

  //std::cout<<"RecHit"<<std::endl;
 
  for(edm::PSimHitContainer::const_iterator itHit = selGEMSimHits.begin(); itHit != selGEMSimHits.end(); ++itHit){

      	GEMDetId idGem = GEMDetId(itHit->detUnitId());
      	int region_sim = idGem.region();
      	int layer_sim = idGem.layer();
      	int chamber_sim = idGem.chamber();
      	int roll_sim = idGem.roll();

      	LocalPoint lp = itHit->entryPoint();
      	float strip_sim = gemGeom->etaPartition(idGem)->strip(lp);

      	if(region != region_sim) continue;
      	if(layer != layer_sim) continue;
      	if(chamber != chamber_sim) continue;
      	if(roll != roll_sim) continue;

      	//if(abs(strip - strip_sim) < 2) result = true;
	for(int i = firstStrip; i < (firstStrip + cls); i++ ){
	
	//std::cout<<"Region: "<<region<<" Layer: "<<layer<<" Chamber: "<<chamber<<" roll "<<roll<<std::endl;
	//std::cout<<"RegionSim: "<<region_sim<<" LayerSim: "<<layer_sim<<" ChamberSim: "<<chamber_sim<<" rollSim "<<roll_sim<<std::endl;
	//std::cout<<"Sim: "<<strip_sim<<" Strip: "<<i<<" Diff: "<<abs(strip_sim - i)<<std::endl;
	if(abs(strip_sim - i) < 1) result = true;

	}

  }

  //std::cout<<"RecHit: "<<result<<std::endl;
  return result;

}

int countDrMatching(SimTrackContainer::const_iterator simTrack, Handle<reco::TrackCollection> staTracks){

	int countMatchingTmp = 0;

	double simEtaTmp = (*simTrack).momentum().eta();
	double simPhiTmp = (*simTrack).momentum().phi();

  	reco::TrackCollection::const_iterator staTrackTmp;

	for (staTrackTmp = staTracks->begin(); staTrackTmp != staTracks->end(); ++staTrackTmp){

		double recPtTmp = staTrackTmp->pt();
		double recEtaTmp = staTrackTmp->momentum().eta();
		double recPhiTmp = staTrackTmp->momentum().phi();
		double dRTmp = sqrt(pow((simEtaTmp-recEtaTmp),2) + pow((simPhiTmp-recPhiTmp),2));

	    	if(!(recPtTmp && abs(recEtaTmp) > 1.64 && abs(recEtaTmp) < 2.1)) continue;
		if(dRTmp > 0.1) continue;
		countMatchingTmp++;

	}

	return countMatchingTmp;

}

struct MyMuon{

	double pt;
	double eta;
	double phi;
	int charge;
	double etaSTA;
	double phiSTA;
	int chargeSTA;
	bool isGlobal;
	bool isTracker;
	bool isStandAlone;	
	int matchedTracks;
	int recHits;
	double globalPt;
	double standAlonePt;
	double trackerPt;
	double dytPt;
	double pickyPt;
	double tpfmsPt;

};

MyMuon muonMatching(const Event & event, SimTrackContainer::const_iterator simTrack, bool NoGem){

	int numMatch = 0;

	double simEta = (*simTrack).momentum().eta();
	double simPhi = (*simTrack).momentum().phi();

  	MyMuon tmpMuon;

	tmpMuon.pt = -999;
	tmpMuon.eta = -999;
	tmpMuon.phi = -999;
	tmpMuon.charge = -999;
	tmpMuon.etaSTA = -999;
	tmpMuon.phiSTA = -999;
	tmpMuon.chargeSTA = -999;
	tmpMuon.recHits = -999;
	tmpMuon.matchedTracks = -999;
	tmpMuon.isGlobal = -999;
	tmpMuon.isTracker = -999;
	tmpMuon.isStandAlone = -999;
	tmpMuon.globalPt = -999;
	tmpMuon.standAlonePt = -999;
	tmpMuon.trackerPt = -999;
	tmpMuon.dytPt = -999;
	tmpMuon.pickyPt = -999;
	tmpMuon.tpfmsPt = -999;

  	Handle<reco::MuonCollection> muons;
  	event.getByLabel("muons", muons);
  	reco::MuonCollection::const_iterator muon;
	for (muon = muons->begin(); muon != muons->end(); ++muon){

		int numGEMRecHits = 0;

		if(!(muon->pt())) continue;
		if(!(abs(muon->eta()) > 1.64 && abs(muon->eta()) < 2.1)) continue;
		double dR = sqrt(pow((simEta-muon->eta()),2) + pow((simPhi-muon->phi()),2));
		if(dR > 0.1) continue;

		tmpMuon.pt = muon->pt();
		tmpMuon.eta = muon->eta();
		tmpMuon.phi = muon->phi();
		tmpMuon.charge = muon->charge();
		tmpMuon.isGlobal = muon->isGlobalMuon();
		tmpMuon.isTracker = muon->isTrackerMuon();
		tmpMuon.isStandAlone = muon->isStandAloneMuon();

		numMatch++;

		TrackRef muonRef = muon->combinedMuon();
		TrackRef trackerRef = muon->innerTrack();
		TrackRef standAloneRef = muon->outerTrack();
		TrackRef pickyRef = muon->pickyTrack();
		TrackRef dytRef = muon->dytTrack();
		TrackRef tpfmsRef = muon->tpfmsTrack();

		if(trackerRef.isNonnull()) tmpMuon.trackerPt = trackerRef->pt();
		if(standAloneRef.isNonnull()){

			tmpMuon.standAlonePt = standAloneRef->pt();
			tmpMuon.etaSTA = standAloneRef->eta();
			tmpMuon.phiSTA = standAloneRef->phi();
			tmpMuon.chargeSTA = standAloneRef->charge();

		}
		if(dytRef.isNonnull()) tmpMuon.dytPt = dytRef->pt();
		if(pickyRef.isNonnull()) tmpMuon.pickyPt = pickyRef->pt();
		if(tpfmsRef.isNonnull()) tmpMuon.tpfmsPt = tpfmsRef->pt();

		if(muonRef.isNonnull()){

			if(muonRef.isNonnull()) tmpMuon.globalPt = muonRef->pt();

			for(trackingRecHit_iterator recHit = muonRef->recHitsBegin(); recHit != muonRef->recHitsEnd(); ++recHit){

				if (!((*recHit)->geographicalId().det() == DetId::Muon)) continue;
				if ((*recHit)->geographicalId().subdetId() == MuonSubdetId::GEM) numGEMRecHits++;

			}

		}

		if(NoGem) numGEMRecHits = 999;
		tmpMuon.recHits = numGEMRecHits;

	}

	tmpMuon.matchedTracks = numMatch;

	return tmpMuon;

}

MyMuon muonTrackMatching(const Event & event, SimTrackContainer::const_iterator simTrack, bool NoGem){

	int numMatch = 0;

	double simEta = (*simTrack).momentum().eta();
	double simPhi = (*simTrack).momentum().phi();

  	MyMuon tmpMuon;
	tmpMuon.pt = -999;
	tmpMuon.eta = -999;
	tmpMuon.phi = -999;
	tmpMuon.charge = -999;
	tmpMuon.etaSTA = -999;
	tmpMuon.phiSTA = -999;
	tmpMuon.chargeSTA = -999;
	tmpMuon.recHits = -999;
	tmpMuon.matchedTracks = -999;
	tmpMuon.isGlobal = -999;
	tmpMuon.isTracker = -999;
	tmpMuon.isStandAlone = -999;
	tmpMuon.globalPt = -999;
	tmpMuon.standAlonePt = -999;
	tmpMuon.trackerPt = -999;
	tmpMuon.dytPt = -999;
	tmpMuon.pickyPt = -999;
	tmpMuon.tpfmsPt = -999;

  	Handle<reco::TrackCollection> staTracks;
  	event.getByLabel(edm::InputTag("globalMuons","","RECO"), staTracks);

  	reco::TrackCollection::const_iterator muon;

	for (muon = staTracks->begin(); muon != staTracks->end(); ++muon){

		int numGEMRecHits = 0;

		if(!(muon->pt())) continue;
		if(!(abs(muon->eta()) > 1.64 && abs(muon->eta()) < 2.1)) continue;
		double dR = sqrt(pow((simEta-muon->eta()),2) + pow((simPhi-muon->phi()),2));
		if(dR > 0.1) continue;

		tmpMuon.pt = muon->pt();
		tmpMuon.eta = muon->eta();
		tmpMuon.phi = muon->phi();
		tmpMuon.charge = muon->charge();

		numMatch++;

		for(trackingRecHit_iterator recHit = muon->recHitsBegin(); recHit != muon->recHitsEnd(); ++recHit){

			if (!((*recHit)->geographicalId().det() == DetId::Muon)) continue;
			if ((*recHit)->geographicalId().subdetId() == MuonSubdetId::GEM) numGEMRecHits++;

		}

		if(NoGem) numGEMRecHits = 999;
		tmpMuon.recHits = numGEMRecHits;

	}

	tmpMuon.matchedTracks = numMatch;

	return tmpMuon;

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

	int countMatching = 0;

	if (abs((*simTrack).type()) != 13) continue;
	if ((*simTrack).noVertex()) continue;
	if ((*simTrack).noGenpart()) continue;

	simEta = (*simTrack).momentum().eta();
	simPhi = (*simTrack).momentum().phi();
	int qGen = simTrack->charge();

	if (abs(simEta) > 2.1 || abs(simEta) < 1.64) continue;

	//std::cout<<"SimEta "<<simEta<<" SimPhi "<<simPhi<<std::endl;

	edm::PSimHitContainer selGEMSimHits = isTrackMatched(simTrack, event, eventSetup);
	int size = selGEMSimHits.size();
	hMatchedSimHits->Fill(size);
	hSimTrackMatch->Fill(size > 0 ? 1 : 0);
	if(size == 0 && noGEMCase_) continue;

	int drMatching = countDrMatching(simTrack, staTracks);
	//std::cout<<"Matching with: "<<drMatching<<" reco tracks"<<std::endl;
	if(drMatching > 1) continue;

	MyMuon muon;
	muon = muonMatching(event, simTrack, noGEMCase_);

	bool muonType = isGlobalMuon_? muon.isGlobal : muon.isStandAlone;
	double muPt = isGlobalMuon_? muon.pt : muon.standAlonePt;
	double muQ = isGlobalMuon_? muon.charge : muon.chargeSTA;

	if(muonType && muon.matchedTracks == 1 && muPt > 0){

		hNumGEMRecHitsMuon->Fill(muon.recHits);

		if(muon.recHits > 0){

			hInvPtResVsPtMuon->Fill(simPt, (muQ/muPt - qGen/simPt)/(qGen/simPt));
			hDeltaChargeMuon->Fill(simPt, qGen-muQ);
			hDRMatchVsPtMuon->Fill(simPt, muon.matchedTracks);
			h1_PresMuon->Fill((muQ/muPt - qGen/simPt)/(qGen/simPt));
			hInvPtResVsEtaMuon->Fill(simEta, (muQ/muPt - qGen/simPt)/(qGen/simPt));
			hDeltaChargeVsEtaMuon->Fill(abs(simEta), qGen-muQ);
			if(qGen * muQ < 0) h1_PresMuon2->Fill((muQ/muPt - qGen/simPt)/(qGen/simPt));

		}

	}
	
	MyMuon globalTrack;
	globalTrack.pt = -999;
	globalTrack.eta = -999;
	globalTrack.phi = -999;
	globalTrack.charge = -999;
	globalTrack.etaSTA = -999;
	globalTrack.phiSTA = -999;
	globalTrack.chargeSTA = -999;
	globalTrack.recHits = -999;
	globalTrack.matchedTracks = -999;
	globalTrack.isGlobal = -999;
	globalTrack.isTracker = -999;
	globalTrack.isStandAlone = -999;
	globalTrack.globalPt = -999;
	globalTrack.standAlonePt = -999;
	globalTrack.trackerPt = -999;
	globalTrack.dytPt = -999;
	globalTrack.pickyPt = -999;
	globalTrack.tpfmsPt = -999;

  	double recPt = 0.;
  	double recPtIP = 0.;

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
	    
	    	if(!(recPt && theDataType == "SimData" && abs(recEta) > 1.64 && abs(recEta) < 2.1)) continue;

		//std::cout<<"SimEta: "<<simEta<<" SimPhi: "<<simPhi<<" SimPt: "<<simPt<<std::endl;
		//std::cout<<"Eta: "<<recEta<<" Phi: "<<recPhi<<" Pt: "<<recPt<<std::endl;

		countMatching++;

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

			if(simPt > 0) hDeltaPhiVsSimTrackPhi2->Fill(phiDegSim, phiDegSim - phiDegSimHit);

		}

		std::vector<bool> collectResults;

		for(trackingRecHit_iterator recHit = staTrack->recHitsBegin(); recHit != staTrack->recHitsEnd(); ++recHit){

			if (!((*recHit)->geographicalId().det() == DetId::Muon)) continue;

			if ((*recHit)->geographicalId().subdetId() == MuonSubdetId::GEM){

				//std::cout<<"GEM id: "<<GEMDetId((*recHit)->geographicalId().rawId())<<std::endl;
				numGEMRecHits++;
				hasGemRecHits = true;

				int index = std::distance(staTrack->recHitsBegin(), recHit);
				TrackingRecHitRef tRH = staTrack->recHit(index);

				bool status = isRecHitMatched(selGEMSimHits, tRH, gemGeom);
				collectResults.push_back(status);

				if(!isGlobalMuon_){

					GEMDetId id((*recHit)->geographicalId());

					int region = id.region();
					int layer = id.layer();
					int chamber = id.chamber();
					int roll = id.roll();

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
						int chamber_sim = idGem.chamber();
						int roll_sim = idGem.roll();
						if (!(region == region_sim && layer == layer_sim && chamber == chamber_sim && roll == roll_sim)) continue;

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

						if(simPt > 0) hDeltaPhiVsSimTrackPhi->Fill(phiDegSim, phiDegSim - phiDegSimHit);

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

		int sizeRH = 0;
		bool matchingHit = true;
		bool matchingParHit = false;
		for(int i = 0; i < (int)collectResults.size(); i++){

			//std::cout<<"Result[i] "<<collectResults[i]<<std::endl;
			if(collectResults[i]) sizeRH++;
			matchingHit &= collectResults[i];
			matchingParHit |= collectResults[i];

		}
		hRecHitMatching->Fill(matchingHit);
		hRecHitParMatching->Fill(matchingParHit);
		hRecoTracksWithMatchedRecHits->Fill(collectResults.size(),sizeRH);
		//std::cout<<"Result "<<matchingHit<<std::endl;

		double simPtCorr = 0;
		if(noGEMCase_){ 

			hasGemRecHits = true;
			matchingHit = true;
			simPtCorr = (recPt - 0.00115)/0.9998;

		}
		else simPtCorr = (recPt - 0.0005451)/0.9999;

		if(hasGemRecHits & /*matchingHit &*/ (includeME11_ ? hasRecHitsFromCSCME11 : !hasRecHitsFromCSCME11)){

			int qRec = staTrack->charge();

			globalTrack.pt = recPt;
			globalTrack.eta = recEta;
			globalTrack.phi = recPhi;
			globalTrack.charge = qRec;
			globalTrack.recHits = collectResults.size();
			globalTrack.matchedTracks = drMatching;

			//TH1::StatOverflows(kTRUE);

			hCharge->Fill(qGen,qRec);
			hDeltaCharge->Fill(simPt,qGen-qRec);
			hDeltaChargeVsEta->Fill(abs(simEta), qGen-qRec);

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
			hDeltaQvsDeltaPt->Fill( ((qRec/recPt - qGen/simPt)/(qGen/simPt)), (qRec-qGen) );
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

			if(muon.pt != -999 && muon.matchedTracks == 1 && muon.pt > 0 && muon.recHits > 0){

				hCountPresence->Fill(1);
				double res1 = abs((recPt-simPt)/simPt);
				double res2 = abs((muon.pt-simPt)/simPt);
				hPtResVsPtRes->Fill(res1,res2);
				hDeltaPtRes->Fill(res1-res2);

			}
			else hCountPresence->Fill(0);

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
    
  	}//Fine loop sulle STA track

	hDRMatchVsPt->Fill(simPt, countMatching);

	if(muon.isGlobal && muon.matchedTracks == 1 && muon.recHits > 0){

		/*std::cout<<"SimTrack: "<<simPt<<" SimEta: "<<simEta
			<<" GLBTrack: "<<setprecision(10)<<globalTrack.pt
			<<" RecoMuon: "<<setprecision(10)<<muon.pt
			<<" RecoMuonGlb: "<<setprecision(10)<<muon.globalPt
			<<" TrkMuon: "<<setprecision(10)<<muon.trackerPt
			<<" Picky: "<<setprecision(10)<<muon.pickyPt
			<<" DYT: "<<setprecision(10)<<muon.dytPt
			<<" TPFMS: "<<setprecision(10)<<muon.tpfmsPt
			<<std::endl;*/

		if(globalTrack.pt == -999 && muon.pt != -999) hCheckGlobalTracksVsPt->Fill(simPt, 0.5);
		else if(globalTrack.pt != -999 && muon.pt == -999) hCheckGlobalTracksVsPt->Fill(simPt, 1.5);
		else if(globalTrack.pt != -999 && muon.pt != -999) hCheckGlobalTracksVsPt->Fill(simPt, 2.5);
		else if(globalTrack.pt == -999 && muon.pt == -999) hCheckGlobalTracksVsPt->Fill(simPt, 3.5);

		if(muon.pt != -999){

   			std::vector<double> residuals;
			residuals.push_back(muon.pt - muon.globalPt);
			residuals.push_back(muon.pt - muon.standAlonePt);
			residuals.push_back(muon.pt - muon.trackerPt);
			residuals.push_back(muon.pt - muon.pickyPt);
			residuals.push_back(muon.pt - muon.dytPt);
			residuals.push_back(muon.pt - muon.tpfmsPt);

			vector<double>::const_iterator it;
			it = min_element(residuals.begin(), residuals.end());
			int idx = it - residuals.begin();

			if(idx == 0) hCheckTracksVsPt->Fill(simPt, 0.5);
			else if(idx == 1) hCheckTracksVsPt->Fill(simPt, 1.5);
			else if(idx == 2) hCheckTracksVsPt->Fill(simPt, 2.5);
			else if(idx == 3) hCheckTracksVsPt->Fill(simPt, 3.5);
			else if(idx == 4) hCheckTracksVsPt->Fill(simPt, 4.5);
			else if(idx == 5) hCheckTracksVsPt->Fill(simPt, 5.5);
			else hCheckTracksVsPt->Fill(simPt, 6.5);

		}

	}

  }//Fine loop sulle SimTrack


  //cout<<"---"<<endl;  
}

DEFINE_FWK_MODULE(STAMuonAnalyzer);
