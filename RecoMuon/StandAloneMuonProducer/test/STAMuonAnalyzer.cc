/** \class STAMuonAnalyzer
 *  Analyzer of the StandAlone muon tracks
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 *  \modified by C. Calabria - INFN Bari
 */

#include "RecoMuon/StandAloneMuonProducer/test/STAMuonAnalyzer.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

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
  muonLabel_ = pset.getUntrackedParameter<edm::InputTag>("MuonCollectionLabel");

  noGEMCase_ = pset.getUntrackedParameter<bool>("NoGEMCase");
  isGlobalMuon_ = pset.getUntrackedParameter<bool>("isGlobalMuon",false);

  minEta_ = pset.getUntrackedParameter<double>("minEta",1.64);
  maxEta_ = pset.getUntrackedParameter<double>("maxEta",2.1);

  theDataType = pset.getUntrackedParameter<string>("DataType");
  
  if(theDataType != "RealData" && theDataType != "SimData")
    cout<<"Error in Data Type!!"<<endl;

  numberOfSimTracks = 0;
  numberOfRecTracks = 0;

}

/// Destructor
STAMuonAnalyzer::~STAMuonAnalyzer(){
}

void STAMuonAnalyzer::beginJob(){

  int bins = 1200;
  int min = -6;
  int max = +6;

  if(isGlobalMuon_){

	bins = 800;
	min = -4;
	max = +4;

  }

  // register to the TFileService
  edm::Service<TFileService> fs;
  TH1::SetDefaultSumw2();

  histContainer_["hPtRec"] = fs->make<TH1F>("pTRec","p_{T}^{rec}",261,-2.5,1302.5);
  histContainer_["hDeltaPtRec"] = fs->make<TH1F>("DeltapTRec","#Delta p_{T}^{rec}",400,-2,2);
  histContainer_["hPtSim"] = fs->make<TH1F>("pTSim","p_{T}^{gen} ",261,-2.5,1302.5);

  histContainer_["hPTDiff"] = fs->make<TH1F>("pTDiff","p_{T}^{rec} - p_{T}^{gen} ",400,-1000,1000);
  histContainer_["hPTDiff2"] = fs->make<TH1F>("pTDiff2","p_{T}^{rec} - p_{T}^{gen} ",400,-1000,1000);

  histContainer2D_["hPTDiffvsEta"] = fs->make<TH2F>("PTDiffvsEta","p_{T}^{rec} - p_{T}^{gen} VS #eta",100,-2.5,2.5,200,-1000,1000);
  histContainer2D_["hPTDiffvsPhi"] = fs->make<TH2F>("PTDiffvsPhi","p_{T}^{rec} - p_{T}^{gen} VS #phi",100,-6,6,200,-1000,1000);

  histContainer_["hPres"] = fs->make<TH1F>("pTRes","pT Resolution",bins,min,max);
  histContainer_["h1_Pres"] = fs->make<TH1F>("invPTRes","1/pT Resolution",bins,min,max);
  histContainer_["h1_PresMuon"] = fs->make<TH1F>("invPTResMuon","1/pT Resolution",1200,-6,6);
  histContainer_["h1_PresMuon2"] = fs->make<TH1F>("invPTResMuon2","1/pT Resolution",1200,-6,6);

  histContainer_["hSimEta"] = fs->make<TH1F>("PSimEta","SimTrack #eta",100,-2.5,2.5);
  histContainer_["hRecEta"] = fs->make<TH1F>("PRecEta","RecTrack #eta",100,-2.5,2.5);
  histContainer_["hDeltaEta"] = fs->make<TH1F>("PDeltaEta","#Delta#eta",100,-1,1);
  histContainer_["hDeltaPhi"] = fs->make<TH1F>("PDeltaPhi","#Delta#phi",400,-2,2);
  histContainer_["hDeltaPhiMuon"] = fs->make<TH1F>("PDeltaPhiMuon","#Delta#phi",400,-2,2);
  histContainer_["hDeltaPhiPlus"] = fs->make<TH1F>("PDeltaPhiPlus","#Delta#phi q>0",400,-2,2);
  histContainer_["hDeltaPhiMinus"] = fs->make<TH1F>("PDeltaPhiMinus","#Delta#phi q<0",400,-2,2);

  histContainer_["hSimPhi"] = fs->make<TH1F>("PSimPhi","SimTrack #phi",100,-TMath::Pi(),TMath::Pi());
  histContainer_["hRecPhi"] = fs->make<TH1F>("PRecPhi","RecTrack #phi",100,-TMath::Pi(),TMath::Pi());
  histContainer2D_["hRecPhi2DPlusLayer1"] = fs->make<TH2F>("RecHitPhi2DPlusLayer1","RecHit #phi GE+1 vs. Sector, Layer 1",36,-TMath::Pi(),TMath::Pi(),36,0,36);
  histContainer2D_["hRecPhi2DMinusLayer1"] = fs->make<TH2F>("RecHitPhi2DMinusLayer1","RecHit #phi GE-1 vs. Sector, Layer 1",36,-TMath::Pi(),TMath::Pi(),36,0,36);
  histContainer2D_["hRecPhi2DPlusLayer2"] = fs->make<TH2F>("RecHitPhi2DPlusLayer2","RecHit #phi GE+1 vs. Sector, Layer 2",36,-TMath::Pi(),TMath::Pi(),36,0,36);
  histContainer2D_["hRecPhi2DMinusLayer2"] = fs->make<TH2F>("RecHitPhi2DMinusLayer2","RecHit #phi GE-1 vs. Sector, Layer 2",36,-TMath::Pi(),TMath::Pi(),36,0,36);

  histContainer_["hNumSimTracks"] = fs->make<TH1F>("NumSimTracks","NumSimTracks",100,0,100);
  histContainer_["hNumMuonSimTracks"] = fs->make<TH1F>("NumMuonSimTracks","NumMuonSimTracks",10,0,10);
  histContainer_["hNumRecTracks"] = fs->make<TH1F>("NumRecTracks","NumRecTracks",10,0,10);

  histContainer_["hNumGEMRecHits"] = fs->make<TH1F>("NumGEMRecHits","NumGEMRecHits",11,-0.5,10.5);
  histContainer_["hNumGEMRecHitsSt1"] = fs->make<TH1F>("NumGEMRecHitsSt1","NumGEMRecHitsSt1",11,-0.5,10.5);
  histContainer_["hNumGEMRecHitsSt2"] = fs->make<TH1F>("NumGEMRecHitsSt2","NumGEMRecHitsSt2",11,-0.5,10.5);
  histContainer_["hNumGEMRecHitsSt3"] = fs->make<TH1F>("NumGEMRecHitsSt3","NumGEMRecHitsSt3",11,-0.5,10.5);
  histContainer_["hNumGEMRecHitsMuon"] = fs->make<TH1F>("NumGEMRecHitsMuon","NumGEMRecHitsMuon",11,-0.5,10.5);

  histContainer2D_["hRecoPtVsSimPt"] = fs->make<TH2F>("RecoPtVsSimPt","p_{T}^{Reco} vs. p_{T}^{Sim}",261,-2.5,1302.5,261,-2.5,1302.5);
  histContainer2D_["hDeltaPtVsSimPt"] = fs->make<TH2F>("DeltaPtVsSimPt","(p_{T}^{Reco} - p_{T}^{Sim}) vs. p_{T}^{Sim}",261,-2.5,1302.5,500,-500,500);

  histContainer2D_["hPtResVsPt"] = fs->make<TH2F>("PtResVsPt","p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,bins,min,max);
  histContainer2D_["hInvPtResVsPt"] = fs->make<TH2F>("InvPtResVsPt","1/p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,bins,min,max);
  histContainer2D_["hInvPtResVsPtMuon"] = fs->make<TH2F>("InvPtResVsPtMuon","1/p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,bins,min,max);
  histContainer2D_["hInvPtResVsPtSelWrong"] = fs->make<TH2F>("InvPtResVsPtSelWrong","1/p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,bins,min,max);
  histContainer2D_["hInvPtResVsPtSelCorr"] = fs->make<TH2F>("InvPtResVsPtSelCorr","1/p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,bins,min,max);
  histContainer2D_["hInvPtResVsPtSelWrongPicky"] = fs->make<TH2F>("InvPtResVsPtSelWrongPicky","1/p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,bins,min,max);
  histContainer2D_["hInvPtResVsPtSelCorrPicky"] = fs->make<TH2F>("InvPtResVsPtSelCorrPicky","1/p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,bins,min,max);
  histContainer2D_["hInvPtResVsPtSelWrongTrk"] = fs->make<TH2F>("InvPtResVsPtSelWrongTrk","1/p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,bins,min,max);
  histContainer2D_["hInvPtResVsPtSelCorrTrk"] = fs->make<TH2F>("InvPtResVsPtSelCorrTrk","1/p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,bins,min,max);

  histContainer2D_["hPtResVsPtNoCharge"] = fs->make<TH2F>("PtResVsPtNoCharge","p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,bins,min,max);
  histContainer2D_["hInvPtResVsPtNoCharge"] = fs->make<TH2F>("InvPtResVsPtNoCharge","1/p_{T} Resolution vs. p_{T}",261,-2.5,1302.5,bins,min,max);

  histContainer2D_["hDPhiVsPt"] = fs->make<TH2F>("DPhiVsPt","#Delta#phi vs. p_{T}",261,-2.5,1302.5,100,-6,6);

  histContainer2D_["hPtResVsEta"] = fs->make<TH2F>("PtResVsEta","p_{T} Resolution vs. #eta",100,-2.5,2.5,bins,min,max);
  histContainer2D_["hInvPtResVsEta"] = fs->make<TH2F>("InvPtResVsEta","1/p_{T} Resolution vs. #eta",100,-2.5,2.5,bins,min,max);
  histContainer2D_["hInvPtResVsEtaMuon"] = fs->make<TH2F>("InvPtResVsEtaMuon","1/p_{T} Resolution vs. #eta",100,-2.5,2.5,bins,min,max);

  histContainer2D_["hPtResVsEtaNoCharge"] = fs->make<TH2F>("PtResVsEtaNoCharge","p_{T} Resolution vs. #eta",100,-2.5,2.5,bins,min,max);
  histContainer2D_["hInvPtResVsEtaNoCharge"] = fs->make<TH2F>("InvPtResVsEtaNoCharge","1/p_{T} Resolution vs. #eta",100,-2.5,2.5,bins,min,max);

  histContainer_["hDenPt"] = fs->make<TH1F>("DenPt","DenPt",261,-2.5,1302.5);
  histContainer_["hDenEta"] = fs->make<TH1F>("DenEta","DenEta",100,-2.5,2.5);
  histContainer_["hDenPhi"] = fs->make<TH1F>("DenPhi","DenPhi",36,-TMath::Pi(),TMath::Pi());
  histContainer_["hDenPhiPlus"] = fs->make<TH1F>("DenPhiPlus","DenPhiMinus",360,0,180);
  histContainer_["hDenPhiMinus"] = fs->make<TH1F>("DenPhiMinus","DenPhiMinus",360,0,180);
  histContainer_["hDenSimPt"] = fs->make<TH1F>("DenSimPt","DenSimPt",261,-2.5,1302.5);
  histContainer_["hDenSimEta"] = fs->make<TH1F>("DenSimEta","DenSimEta",100,-2.5,2.5);
  histContainer_["hDenSimPhiPlus"] = fs->make<TH1F>("DenSimPhiPlus","DenSimPhiMinus",360,0,180);
  histContainer_["hDenSimPhiMinus"] = fs->make<TH1F>("DenSimPhiMinus","DenSimPhiMinus",360,0,180);

  histContainer_["hNumPt"] = fs->make<TH1F>("NumPt","NumPt",261,-2.5,1302.5);
  histContainer_["hNumEta"] = fs->make<TH1F>("NumEta","NumEta",100,-2.5,2.5);
  histContainer_["hNumPhi"] = fs->make<TH1F>("NumPhi","NumPhi",36,-TMath::Pi(),TMath::Pi());
  histContainer_["hNumPhiPlus"] = fs->make<TH1F>("NumPhiPlus","NumPhiMinus",360,0,180);
  histContainer_["hNumPhiMinus"] = fs->make<TH1F>("NumPhiMinus","NumPhiMinus",360,0,180);
  histContainer_["hNumSimPt"] = fs->make<TH1F>("NumSimPt","NumSimPt",261,-2.5,1302.5);
  histContainer_["hNumSimEta"] = fs->make<TH1F>("NumSimEta","NumSimEta",100,-2.5,2.5);
  histContainer_["hNumSimPhiPlus"] = fs->make<TH1F>("NumSimPhiPlus","NumSimPhiMinus",360,0,180);
  histContainer_["hNumSimPhiMinus"] = fs->make<TH1F>("NumSimPhiMinus","NumSimPhiMinus",360,0,180);

  histContainer_["hNumSimPtSt1"] = fs->make<TH1F>("NumSimPtSt1","NumSimPtSt1",261,-2.5,1302.5);
  histContainer_["hNumSimEtaSt1"] = fs->make<TH1F>("NumSimEtaSt1","NumSimEtaSt1",100,-2.5,2.5);
  histContainer_["hNumSimPhiPlusSt1"] = fs->make<TH1F>("NumSimPhiPlusSt1","NumSimPhiMinusSt1",360,0,180);
  histContainer_["hNumSimPhiMinusSt1"] = fs->make<TH1F>("NumSimPhiMinusSt1","NumSimPhiMinusSt1",360,0,180);
  histContainer_["hNumSimPtSt2"] = fs->make<TH1F>("NumSimPtSt2","NumSimPtSt2",261,-2.5,1302.5);
  histContainer_["hNumSimEtaSt2"] = fs->make<TH1F>("NumSimEtaSt2","NumSimEtaSt2",100,-2.5,2.5);
  histContainer_["hNumSimPhiPlusSt2"] = fs->make<TH1F>("NumSimPhiPlusSt2","NumSimPhiMinusSt2",360,0,180);
  histContainer_["hNumSimPhiMinusSt2"] = fs->make<TH1F>("NumSimPhiMinusSt2","NumSimPhiMinusSt2",360,0,180);
  histContainer_["hNumSimPtSt3"] = fs->make<TH1F>("NumSimPtSt3","NumSimPtSt3",261,-2.5,1302.5);
  histContainer_["hNumSimEtaSt3"] = fs->make<TH1F>("NumSimEtaSt3","NumSimEtaSt3",100,-2.5,2.5);
  histContainer_["hNumSimPhiPlusSt3"] = fs->make<TH1F>("NumSimPhiPlusSt3","NumSimPhiMinusSt3",360,0,180);
  histContainer_["hNumSimPhiMinusSt3"] = fs->make<TH1F>("NumSimPhiMinusSt3","NumSimPhiMinusSt3",360,0,180);

  histContainer_["hPullGEMx"] = fs->make<TH1F>("PullGEMx", "(x_{mc} - x_{rec}) / #sigma",500,-10.,10.);
  histContainer_["hPullGEMphi"] = fs->make<TH1F>("PullGEMphi", "(#phi_{mc} - #phi_{rec})",500,-0.001,0.001);

  histContainer_["hGEMRecHitEta"] = fs->make<TH1F>("GEMRecHitEta","GEM RecHits #eta",10000,-2.5,2.5);
  histContainer_["hGEMRecHitEtaSt1L1"] = fs->make<TH1F>("hGEMRecHitEtaSt1L1","GEM RecHits #eta",10000,-2.5,2.5);
  histContainer_["hGEMRecHitEtaSt1L2"] = fs->make<TH1F>("hGEMRecHitEtaSt1L2","GEM RecHits #eta",10000,-2.5,2.5);
  histContainer_["hGEMRecHitEtaSt2L1"] = fs->make<TH1F>("hGEMRecHitEtaSt2L1","GEM RecHits #eta",10000,-2.5,2.5);
  histContainer_["hGEMRecHitEtaSt2L2"] = fs->make<TH1F>("hGEMRecHitEtaSt2L2","GEM RecHits #eta",10000,-2.5,2.5);
  histContainer_["hGEMRecHitEtaSt3L1"] = fs->make<TH1F>("hGEMRecHitEtaSt3L1","GEM RecHits #eta",10000,-2.5,2.5);
  histContainer_["hGEMRecHitEtaSt3L2"] = fs->make<TH1F>("hGEMRecHitEtaSt3L2","GEM RecHits #eta",10000,-2.5,2.5);
  histContainer_["hGEMRecHitPhi"] = fs->make<TH1F>("GEMRecHitPhi","GEM RecHits #phi",360,-TMath::Pi(),TMath::Pi());

  histContainer_["hDR"] = fs->make<TH1F>("DR","#Delta R (SIM-RECO)",300,0,1);

  histContainer2D_["hCharge"] = fs->make<TH2F>("Charge","q (SIM-RECO)",6,-3,3,6,-3,3);
  histContainer2D_["hDeltaCharge"] = fs->make<TH2F>("DeltaCharge","#Delta q (SIM-RECO)",261,-2.5,1302.5,6,-3,3);
  histContainer2D_["hDeltaChargeMuon"] = fs->make<TH2F>("DeltaChargeMuon","#Delta q (SIM-RECO)",261,-2.5,1302.5,6,-3,3);
  histContainer2D_["hDeltaChargeVsEta"] = fs->make<TH2F>("DeltaChargeVsEta","#Delta q (SIM-RECO) vs. #eta",50,0,2.5,6,-3,3);
  histContainer2D_["hDeltaChargeVsEtaMuon"] = fs->make<TH2F>("DeltaChargeVsEtaMuon","#Delta q (SIM-RECO) vs. #eta",50,0,2.5,6,-3,3);

  histContainer2D_["hDeltaPhiVsSimTrackPhi"] = fs->make<TH2F>("DeltaPhiVsSimTrackPhi","DeltaPhiVsSimTrackPhi",360,0,180,2000,-20,+20);

  histContainer_["hSimTrackMatch"] = fs->make<TH1F>("SimTrackMatch", "SimTrackMatch",2,0.,2.);
  histContainer_["hRecHitMatching"] = fs->make<TH1F>("RecHitMatching", "RecHitMatching",2,0.,2.);
  histContainer_["hRecHitParMatching"] = fs->make<TH1F>("RecHitParMatching", "RecHitParMatching",2,0.,2.);
  histContainer2D_["hDRMatchVsPt"] = fs->make<TH2F>("DRMatchVsPt","DRMatchVsPt",261,-2.5,1302.5,10,0,10);
  histContainer2D_["hDRMatchVsPtMuon"] = fs->make<TH2F>("DRMatchVsPtMuon","DRMatchVsPtMupn",261,-2.5,1302.5,10,0,10);

  histContainer_["hMatchedSimHits"] = fs->make<TH1F>("MatchedSimHits","MatchedSimHits",6,-0.5,5.5);
  histContainer2D_["hRecoTracksWithMatchedRecHits"] = fs->make<TH2F>("RecoTracksWithMatchedRecHits","RecoTracksWithMatchedRecHits",6,-0.5,5.5,6,-0.5,5.5);
  histContainer2D_["hDeltaQvsDeltaPt"] = fs->make<TH2F>("DeltaQvsDeltaPt","DeltaQvsDeltaPt",100,-2,2,7,-3.5,3.5);
  histContainer2D_["hCheckGlobalTracksVsPt"] = fs->make<TH2F>("CheckGlobalTracksVsPt","CheckGlobalTracksVsPt",261,-2.5,1302.5,4,0,4);
  histContainer2D_["hCheckTracksVsPt"] = fs->make<TH2F>("CheckTracksVsPt","CheckTracksVsPt",261,-2.5,1302.5,8,0,8);
  histContainer2D_["hCheckChargeVsPt"] = fs->make<TH2F>("CheckChargeVsPt","CheckChargeVsPt",261,-2.5,1302.5,8,0,8);

  histContainer2D_["hPtResVsPtRes"] = fs->make<TH2F>("PtResVsPtRes","PtResVsPtRes",400,-2,2,400,-2,2);
  histContainer_["hDeltaPtRes"] = fs->make<TH1F>("DeltaPtRes","DeltaPtRes",400,-2,2);

  histContainer_["hCountPresence"] = fs->make<TH1F>("CountPresence","CountPresence",2,0,2);

}

void STAMuonAnalyzer::endJob(){

  if(theDataType == "SimData"){
    cout << endl << endl << "Number of Sim tracks: " << numberOfSimTracks << endl;
  }

  cout << "Number of Reco tracks: " << numberOfRecTracks << endl << endl;

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

struct MyGEMSimHit
{  

  	int particleType;
  	float x, y;
  	int region, ring, station, layer, chamber, roll;
  	float globalEta, globalPhi, globalX, globalY, globalZ;
  	int strip;

};

struct MyGEMRecHit
{

  	float x, y, xErr;
  	int region, ring, station, layer, chamber, roll;
  	float globalEta, globalPhi, globalX, globalY, globalZ;
  	int bx, clusterSize, firstClusterStrip;

};


bool isRecHitMatched(edm::PSimHitContainer selGEMSimHits, TrackingRecHitRef recHit, edm::ESHandle<GEMGeometry> gemGeom, MyGEMSimHit &sh, MyGEMRecHit &rh)
{

  bool result = false;

  GEMDetId id(recHit->geographicalId());
  LocalPoint lp1 = recHit->localPosition();

  int region = id.region();
  int layer = id.layer();
  int station = id.station();
  int chamber = id.chamber();
  int roll = id.roll();
  
  const GEMRecHit * gemrechit = dynamic_cast<const GEMRecHit *>(&(*recHit));
  GlobalPoint hitGP = gemGeom->idToDet(gemrechit->gemId())->surface().toGlobal(lp1);
  int cls = gemrechit->clusterSize();
  int firstStrip = gemrechit->firstClusterStrip();
  int bx = gemrechit->BunchX();

  //std::cout<<"RecHit"<<std::endl;
 
  for(edm::PSimHitContainer::const_iterator itHit = selGEMSimHits.begin(); itHit != selGEMSimHits.end(); ++itHit){

      	GEMDetId idGem = GEMDetId(itHit->detUnitId());
      	int region_sim = idGem.region();
      	int layer_sim = idGem.layer();
      	int station_sim = idGem.station();
      	int chamber_sim = idGem.chamber();
      	int roll_sim = idGem.roll();

      	LocalPoint lp = itHit->entryPoint();
        GlobalPoint hitGP_sim(gemGeom->idToDet(itHit->detUnitId())->surface().toGlobal(lp));
      	float strip_sim = gemGeom->etaPartition(idGem)->strip(lp);

      	if(region != region_sim) continue;
      	if(layer != layer_sim) continue;
      	if(station != station_sim) continue;
      	if(chamber != chamber_sim) continue;
      	if(roll != roll_sim) continue;

	for(int i = firstStrip; i < (firstStrip + cls); i++ ){
	
		//std::cout<<"Sim: "<<strip_sim<<" Strip: "<<i<<" Diff: "<<abs(strip_sim - i)<<std::endl;
		if(abs(strip_sim - i) < 1) result = true;
		if(!result) continue;
		//std::cout<<"Region: "<<region<<" Layer: "<<layer<<" Chamber: "<<chamber<<" roll "<<roll<<std::endl;
		//std::cout<<"RegionSim: "<<region_sim<<" LayerSim: "<<layer_sim<<" ChamberSim: "<<chamber_sim<<" rollSim "<<roll_sim<<std::endl;
		//std::cout<<"RecHitPhi "<<hitGP.phi()<<" SimHitPhi "<<hitGP_sim.phi()<<std::endl;
		
	}

	if(result){

		sh.particleType = itHit->particleType();
		sh.region = region_sim;
		sh.layer = layer_sim;
		sh.station = station_sim;
		sh.chamber = chamber_sim;
		sh.roll = roll_sim;
		sh.strip = strip_sim;
		sh.x = lp.x();
		sh.y = lp.y();
		sh.globalEta = hitGP_sim.eta();
		sh.globalPhi = hitGP_sim.phi();
		sh.globalX = hitGP_sim.x();
		sh.globalY = hitGP_sim.y();
		sh.globalZ = hitGP_sim.z();

		rh.region = region;
		rh.layer = layer;
		rh.station = station;
		rh.chamber = chamber;
		rh.roll = roll_sim;
		rh.bx = bx;
		rh.firstClusterStrip = firstStrip;
		rh.clusterSize = cls;
		rh.x = lp1.x();
		rh.xErr = gemrechit->localPositionError().xx();
		rh.y = lp1.y();
		rh.globalEta = hitGP.eta();
		rh.globalPhi = hitGP.phi();
		rh.globalX = hitGP.x();
		rh.globalY = hitGP.y();
		rh.globalZ = hitGP.z();

	}

  }

  //std::cout<<"RecHit: "<<result<<std::endl;
  return result;

}

int countDrMatching(SimTrackContainer::const_iterator simTrack, Handle<reco::TrackCollection> staTracks, float minEta_, float maxEta_){

	int countMatchingTmp = 0;

	double simEtaTmp = (*simTrack).momentum().eta();
	double simPhiTmp = (*simTrack).momentum().phi();

  	reco::TrackCollection::const_iterator staTrackTmp;

	for (staTrackTmp = staTracks->begin(); staTrackTmp != staTracks->end(); ++staTrackTmp){

		double recPtTmp = staTrackTmp->pt();
		double recEtaTmp = staTrackTmp->momentum().eta();
		double recPhiTmp = staTrackTmp->momentum().phi();
		double dRTmp = sqrt(pow((simEtaTmp-recEtaTmp),2) + pow((simPhiTmp-recPhiTmp),2));

	    	if(!(recPtTmp && abs(recEtaTmp) > minEta_ && abs(recEtaTmp) < maxEta_)) continue;
		if(dRTmp > 0.1) continue;
		countMatchingTmp++;

	}

	return countMatchingTmp;

}

struct MyMuon{

	double pt, eta, phi;
	int charge;
	double etaSTA, phiSTA;
	int chargeSTA;
	bool isGlobal, isTracker, isStandAlone;	
	int matchedTracks, recHits;
	double globalPt, standAlonePt, trackerPt;
	double dytPt, pickyPt, tpfmsPt;

};

MyMuon muonMatching(const Event & event, SimTrackContainer::const_iterator simTrack, bool NoGem, float minEta_, float maxEta_, edm::InputTag muonLabel_){

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
  	event.getByLabel(muonLabel_, muons);
  	reco::MuonCollection::const_iterator muon;
	for (muon = muons->begin(); muon != muons->end(); ++muon){

		int numGEMRecHits = 0;

		if(!(muon->pt())) continue;
		if(!(abs(muon->eta()) > minEta_ && abs(muon->eta()) < maxEta_)) continue;
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

MyMuon muonTrackMatching(const Event & event, SimTrackContainer::const_iterator simTrack, bool NoGem, float minEta_, float maxEta_){

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
		if(!(abs(muon->eta()) > minEta_ && abs(muon->eta()) < maxEta_)) continue;
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

  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  eventSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

  edm::Handle<GEMRecHitCollection> gemRecHits; 
  event.getByLabel("gemRecHits","",gemRecHits);

  edm::Handle<edm::PSimHitContainer> GEMHits;
  event.getByLabel(edm::InputTag("g4SimHits","MuonGEMHits"), GEMHits);

  double simPt = 0.;
  double simEta = 0.;
  double simPhi = 0.;

  histContainer_["hNumRecTracks"]->Fill(staTracks->size());

  int simCount = 0;
  //int numRecoTrack = 0;

  // Get the SimTrack collection from the event
  if(theDataType == "SimData"){

	//numRecoTrack = simTracks->size();
  	histContainer_["hNumSimTracks"]->Fill(simTracks->size());
    
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
	histContainer_["hSimEta"]->Fill((*simTrack).momentum().eta());
	histContainer_["hSimPhi"]->Fill((*simTrack).momentum().phi());	

    	}

	histContainer_["hNumMuonSimTracks"]->Fill(simCount);
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

	if (abs(simEta) > maxEta_ || abs(simEta) < minEta_) continue;

	//std::cout<<"SimEta "<<simEta<<" SimPhi "<<simPhi<<std::endl;

	edm::PSimHitContainer selGEMSimHits = isTrackMatched(simTrack, event, eventSetup);
	int size = selGEMSimHits.size();
	histContainer_["hMatchedSimHits"]->Fill(size);
	histContainer_["hSimTrackMatch"]->Fill(size > 0 ? 1 : 0);
	if(noGEMCase_) size = 1;
	if(size == 0) continue;

	int drMatching = countDrMatching(simTrack, staTracks, minEta_, maxEta_);
	//std::cout<<"Matching with: "<<drMatching<<" reco tracks"<<std::endl;
	if(drMatching > 1) continue;

	MyMuon muon;
	muon = muonMatching(event, simTrack, noGEMCase_, minEta_, maxEta_, muonLabel_);

	bool muonType = isGlobalMuon_? muon.isGlobal : muon.isStandAlone;
	double muPt = isGlobalMuon_? muon.pt : muon.standAlonePt;
	double muQ = isGlobalMuon_? muon.charge : muon.chargeSTA;

	if(muonType && muon.matchedTracks == 1 && muPt > 0){

		histContainer_["hNumGEMRecHitsMuon"]->Fill(muon.recHits);

		if(muon.recHits > 0){

			histContainer2D_["hInvPtResVsPtMuon"]->Fill(simPt, (muQ/muPt - qGen/simPt)/(qGen/simPt));
			histContainer2D_["hDeltaChargeMuon"]->Fill(simPt, qGen-muQ);
			histContainer2D_["hDRMatchVsPtMuon"]->Fill(simPt, muon.matchedTracks);
			histContainer_["h1_PresMuon"]->Fill((muQ/muPt - qGen/simPt)/(qGen/simPt));
			histContainer2D_["hInvPtResVsEtaMuon"]->Fill(simEta, (muQ/muPt - qGen/simPt)/(qGen/simPt));
			histContainer2D_["hDeltaChargeVsEtaMuon"]->Fill(abs(simEta), qGen-muQ);
			if(qGen * muQ < 0) histContainer_["h1_PresMuon2"]->Fill((muQ/muPt - qGen/simPt)/(qGen/simPt));

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
	    
	    	histContainer_["hPtRec"]->Fill(recPt);
	    	histContainer_["hDeltaPtRec"]->Fill(recPt - recPtIP);
	    
	    	if(!(recPt && theDataType == "SimData" && abs(recEta) > minEta_ && abs(recEta) < maxEta_)) continue;

		//std::cout<<"SimEta: "<<simEta<<" SimPhi: "<<simPhi<<" SimPt: "<<simPt<<std::endl;
		//std::cout<<"Eta: "<<recEta<<" Phi: "<<recPhi<<" Pt: "<<recPt<<std::endl;

		countMatching++;

		bool hasGemRecHits = false;
		int numGEMRecHits = 0, numGEMRecHitsSt1 = 0, numGEMRecHitsSt2 = 0, numGEMRecHitsSt3 = 0;

		std::vector<bool> collectResults;

		float phi_02pi = recPhi < 0 ? recPhi + TMath::Pi() : recPhi;
		float phiDeg = phi_02pi * 180/ TMath::Pi();
		float phi_02pi_sim = simPhi < 0 ? simPhi + TMath::Pi() : simPhi;
		float phiDegSim = phi_02pi_sim * 180/ TMath::Pi();

		for(trackingRecHit_iterator recHit = staTrack->recHitsBegin(); recHit != staTrack->recHitsEnd(); ++recHit){

			if (!((*recHit)->geographicalId().det() == DetId::Muon)) continue;

			if ((*recHit)->geographicalId().subdetId() == MuonSubdetId::GEM){

				//std::cout<<"GEM id: "<<GEMDetId((*recHit)->geographicalId().rawId())<<std::endl;
				numGEMRecHits++;
				hasGemRecHits = true;

				int index = std::distance(staTrack->recHitsBegin(), recHit);
				TrackingRecHitRef tRH = staTrack->recHit(index);

				MyGEMSimHit sh;
				MyGEMRecHit rh;
				bool status = isRecHitMatched(selGEMSimHits, tRH, gemGeom, sh, rh);
				collectResults.push_back(status);

				if(rh.station == 1) numGEMRecHitsSt1++;
				if(rh.station == 2) numGEMRecHitsSt2++;
				if(rh.station == 3) numGEMRecHitsSt3++;

				if(!isGlobalMuon_ & status){
				
					histContainer_["hGEMRecHitEta"]->Fill(rh.globalEta);

					if(rh.station == 1 && rh.layer == 1) histContainer_["hGEMRecHitEtaSt1L1"]->Fill(rh.globalEta);
					if(rh.station == 1 && rh.layer == 2) histContainer_["hGEMRecHitEtaSt1L2"]->Fill(rh.globalEta);
					if(rh.station == 2 && rh.layer == 1) histContainer_["hGEMRecHitEtaSt2L1"]->Fill(rh.globalEta);
					if(rh.station == 2 && rh.layer == 2) histContainer_["hGEMRecHitEtaSt2L2"]->Fill(rh.globalEta);
					if(rh.station == 3 && rh.layer == 1) histContainer_["hGEMRecHitEtaSt3L1"]->Fill(rh.globalEta);
					if(rh.station == 3 && rh.layer == 2) histContainer_["hGEMRecHitEtaSt3L2"]->Fill(rh.globalEta);

					histContainer_["hGEMRecHitPhi"]->Fill(rh.globalPhi);
					if(rh.region > 0 && rh.layer == 1) histContainer2D_["hRecPhi2DPlusLayer1"]->Fill(rh.globalPhi, rh.chamber);
					else if(rh.region > 0 && rh.layer == 2) histContainer2D_["hRecPhi2DPlusLayer2"]->Fill(rh.globalPhi, rh.chamber);
					else if(rh.region < 0 && rh.layer == 1) histContainer2D_["hRecPhi2DMinusLayer1"]->Fill(rh.globalPhi, rh.chamber);
					else if(rh.region < 0 && rh.layer == 2) histContainer2D_["hRecPhi2DMinusLayer2"]->Fill(rh.globalPhi, rh.chamber);
					//std::cout<<"GEMRecHitEta "<<rh.globalEta<<std::endl;
					//std::cout<<"GEMRecHitPhi "<<rh.globalPhi<<" GEMSimHitPhi"<<sh.globalPhi<<std::endl;
				
					float x_sim = sh.x;
					float x_reco = rh.x;
					float err_x_reco = rh.xErr;
					float dX = x_sim - x_reco;
					float pullX = dX/std::sqrt(err_x_reco);
					histContainer_["hPullGEMx"]->Fill(pullX);

					float phi_sim = sh.globalPhi;
					float phi_reco = rh.globalPhi;
					float dPhi = phi_reco - phi_sim;
					histContainer_["hPullGEMphi"]->Fill(dPhi);

					float phi_02pi_simHit = sh.globalPhi < 0 ? sh.globalPhi + TMath::Pi() : sh.globalPhi;
					float phiDegSimHit = phi_02pi_simHit * 180/ TMath::Pi();

					if(simPt > 0) histContainer2D_["hDeltaPhiVsSimTrackPhi"]->Fill(phiDegSim, phiDegSim - phiDegSimHit);

				}

			}

		}

		histContainer_["hNumGEMRecHits"]->Fill(numGEMRecHits);
		histContainer_["hNumGEMRecHitsSt1"]->Fill(numGEMRecHitsSt1);
		histContainer_["hNumGEMRecHitsSt2"]->Fill(numGEMRecHitsSt2);
		histContainer_["hNumGEMRecHitsSt3"]->Fill(numGEMRecHitsSt3);

		int sizeRH = 0;
		bool matchingHit = true;
		bool matchingParHit = false;
		for(int i = 0; i < (int)collectResults.size(); i++){

			//std::cout<<"Result[i] "<<collectResults[i]<<std::endl;
			if(collectResults[i]) sizeRH++;
			matchingHit &= collectResults[i];
			matchingParHit |= collectResults[i];

		}
		histContainer_["hRecHitMatching"]->Fill(matchingHit);
		histContainer_["hRecHitParMatching"]->Fill(matchingParHit);
		histContainer2D_["hRecoTracksWithMatchedRecHits"]->Fill(collectResults.size(),sizeRH);
		//std::cout<<"Result "<<matchingHit<<std::endl;

		if(noGEMCase_) hasGemRecHits = true;

		if(hasGemRecHits /*& matchingHit*/){

			int qRec = staTrack->charge();

			globalTrack.pt = recPt;
			globalTrack.eta = recEta;
			globalTrack.phi = recPhi;
			globalTrack.charge = qRec;
			globalTrack.recHits = collectResults.size();
			globalTrack.matchedTracks = drMatching;

			//TH1::StatOverflows(kTRUE);

			histContainer2D_["hCharge"]->Fill(qGen, qRec);
			histContainer2D_["hDeltaCharge"]->Fill(simPt, qGen-qRec);
			histContainer2D_["hDeltaChargeVsEta"]->Fill(abs(simEta), qGen-qRec);

			//cout<<"RecEta "<<recEta<<" recPhi "<<recPhi<<std::endl;
			//cout<<"SimEta "<<simEta<<" SimPhi "<<simPhi<<std::endl;
			//cout<<"dR "<<dR<<std::endl;

			histContainer_["hDR"]->Fill(dR);
			histContainer2D_["hRecoPtVsSimPt"]->Fill(simPt, recPt);
			histContainer2D_["hDeltaPtVsSimPt"]->Fill(simPt, recPt - simPt);

			histContainer_["hPres"]->Fill((recPt-simPt)/simPt);

			histContainer2D_["hPtResVsPt"]->Fill(simPt, (recPt*qRec-simPt*qGen)/(simPt*qGen));
			histContainer2D_["hPtResVsEta"]->Fill(simEta, (recPt*qRec-simPt*qGen)/(simPt*qGen));

			histContainer2D_["hPtResVsPtNoCharge"]->Fill(simPt, (recPt-simPt)/simPt);
			histContainer2D_["hPtResVsEtaNoCharge"]->Fill(simEta, (recPt-simPt)/simPt);

			histContainer_["hPtSim"]->Fill(simPt);

			histContainer_["hPTDiff"]->Fill(recPt-simPt);
			histContainer_["hRecEta"]->Fill(recEta);

			histContainer_["hDeltaEta"]->Fill(simEta - recEta);
			histContainer_["hDeltaPhi"]->Fill(simPhi - recPhi);
			if(track.charge() > 0) histContainer_["hDeltaPhiPlus"]->Fill(simPhi - recPhi);
			else if(track.charge() < 0) histContainer_["hDeltaPhiMinus"]->Fill(simPhi - recPhi);

			histContainer_["hRecPhi"]->Fill(recPhi);
			histContainer_["hPTDiff2"]->Fill(track.innermostMeasurementState().globalMomentum().perp()-simPt);
			histContainer2D_["hPTDiffvsEta"]->Fill(recEta,recPt-simPt);
			histContainer2D_["hPTDiffvsPhi"]->Fill(recPhi,recPt-simPt);

			//if( ((recPt-simPt)/simPt) <= -0.4)
			//	cout<<"Out of Res: "<<(recPt-simPt)/simPt<<endl;

			histContainer_["h1_Pres"]->Fill((qRec/recPt - qGen/simPt)/(qGen/simPt));

			histContainer2D_["hInvPtResVsPtNoCharge"]->Fill(simPt, (1/recPt - 1/simPt)/(1/simPt));
			histContainer2D_["hInvPtResVsEtaNoCharge"]->Fill(simEta, (1/recPt - 1/simPt)/(1/simPt));

			histContainer2D_["hInvPtResVsPt"]->Fill(simPt, (qRec/recPt - qGen/simPt)/(qGen/simPt));
			histContainer2D_["hDeltaQvsDeltaPt"]->Fill( ((qRec/recPt - qGen/simPt)/(qGen/simPt)), (qRec-qGen) );
			histContainer2D_["hInvPtResVsEta"]->Fill(simEta, (qRec/recPt - qGen/simPt)/(qGen/simPt));

			histContainer2D_["hDPhiVsPt"]->Fill(simPt, recPhi-simPhi);

			histContainer_["hNumPt"]->Fill(recPt);
			histContainer_["hNumEta"]->Fill(recEta);
			histContainer_["hNumSimPt"]->Fill(simPt);
			histContainer_["hNumSimEta"]->Fill(simEta);
			histContainer_["hNumPhi"]->Fill(phi_02pi);

			if(recEta > 0) histContainer_["hNumPhiPlus"]->Fill(phiDeg);
			else if(recEta < 0) histContainer_["hNumPhiMinus"]->Fill(phiDeg);
			if(simEta > 0) histContainer_["hNumSimPhiPlus"]->Fill(phiDegSim);
			else if(simEta < 0) histContainer_["hNumSimPhiMinus"]->Fill(phiDegSim);

			if(numGEMRecHitsSt1 > 0){

				histContainer_["hNumSimPtSt1"]->Fill(simPt);
				histContainer_["hNumSimEtaSt1"]->Fill(simEta);
				if(simEta > 0) histContainer_["hNumSimPhiPlusSt1"]->Fill(phiDegSim);
				else if(simEta < 0) histContainer_["hNumSimPhiMinusSt1"]->Fill(phiDegSim);

			}
			if(numGEMRecHitsSt2 > 0){

				histContainer_["hNumSimPtSt2"]->Fill(simPt);
				histContainer_["hNumSimEtaSt2"]->Fill(simEta);
				if(simEta > 0) histContainer_["hNumSimPhiPlusSt2"]->Fill(phiDegSim);
				else if(simEta < 0) histContainer_["hNumSimPhiMinusSt2"]->Fill(phiDegSim);

			}
			if(numGEMRecHitsSt3 > 0){

				histContainer_["hNumSimPtSt3"]->Fill(simPt);
				histContainer_["hNumSimEtaSt3"]->Fill(simEta);
				if(simEta > 0) histContainer_["hNumSimPhiPlusSt3"]->Fill(phiDegSim);
				else if(simEta < 0) histContainer_["hNumSimPhiMinusSt3"]->Fill(phiDegSim);

			}

			if(muon.matchedTracks == 1 && muon.pt > 0 && muon.recHits > 0){

				histContainer_["hCountPresence"]->Fill(1);
				double res1 = abs((recPt-simPt)/simPt);
				double res2 = abs((muon.pt-simPt)/simPt);
				histContainer2D_["hPtResVsPtRes"]->Fill(res1,res2);
				histContainer_["hDeltaPtRes"]->Fill(res1-res2);

			}
			else histContainer_["hCountPresence"]->Fill(0);

		}

		histContainer_["hDenPt"]->Fill(recPt);
		histContainer_["hDenEta"]->Fill(recEta);
		histContainer_["hDenSimPt"]->Fill(simPt);
		histContainer_["hDenSimEta"]->Fill(simEta);

		histContainer_["hDenPhi"]->Fill(phi_02pi);
		if(recEta > 0) histContainer_["hDenPhiPlus"]->Fill(phiDeg);
		else if(recEta < 0) histContainer_["hDenPhiMinus"]->Fill(phiDeg);
		if(simEta > 0) histContainer_["hDenSimPhiPlus"]->Fill(phiDegSim);
		else if(simEta < 0) histContainer_["hDenSimPhiMinus"]->Fill(phiDegSim);
    
  	}//Fine loop sulle STA track

	histContainer2D_["hDRMatchVsPt"]->Fill(simPt, countMatching);

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

		if(globalTrack.pt == -999 && muon.pt != -999) histContainer2D_["hCheckGlobalTracksVsPt"]->Fill(simPt, 0.5);
		else if(globalTrack.pt != -999 && muon.pt == -999) histContainer2D_["hCheckGlobalTracksVsPt"]->Fill(simPt, 1.5);
		else if(globalTrack.pt != -999 && muon.pt != -999) histContainer2D_["hCheckGlobalTracksVsPt"]->Fill(simPt, 2.5);
		else if(globalTrack.pt == -999 && muon.pt == -999) histContainer2D_["hCheckGlobalTracksVsPt"]->Fill(simPt, 3.5);

		if(muon.pt != -999){

   			std::vector<double> residuals;
			residuals.push_back(abs(muon.pt - muon.globalPt));
			residuals.push_back(abs(muon.pt - muon.standAlonePt));
			residuals.push_back(abs(muon.pt - muon.trackerPt));
			residuals.push_back(abs(muon.pt - muon.pickyPt));
			residuals.push_back(abs(muon.pt - muon.dytPt));
			residuals.push_back(abs(muon.pt - muon.tpfmsPt));

			vector<double>::const_iterator it;
			it = min_element(residuals.begin(), residuals.end());
			int idx = it - residuals.begin();

			if(idx == 0 && muQ*qGen < 0) histContainer2D_["hInvPtResVsPtSelWrong"]->Fill(simPt,(muQ/muPt - qGen/simPt)/(qGen/simPt));
			else if(idx == 0 && muQ*qGen > 0) histContainer2D_["hInvPtResVsPtSelCorr"]->Fill(simPt,(muQ/muPt - qGen/simPt)/(qGen/simPt));

			if(idx == 3 && muQ*qGen < 0) histContainer2D_["hInvPtResVsPtSelWrongPicky"]->Fill(simPt,(muQ/muPt - qGen/simPt)/(qGen/simPt));
			else if(idx == 3 && muQ*qGen > 0) histContainer2D_["hInvPtResVsPtSelCorrPicky"]->Fill(simPt,(muQ/muPt - qGen/simPt)/(qGen/simPt));

			if(idx == 2 && muQ*qGen < 0) histContainer2D_["hInvPtResVsPtSelWrongTrk"]->Fill(simPt,(muQ/muPt - qGen/simPt)/(qGen/simPt));
			else if(idx == 2 && muQ*qGen > 0) histContainer2D_["hInvPtResVsPtSelCorrTrk"]->Fill(simPt,(muQ/muPt - qGen/simPt)/(qGen/simPt));

			if(idx == 0) histContainer2D_["hCheckTracksVsPt"]->Fill(simPt, 0.5);
			else if(idx == 1) histContainer2D_["hCheckTracksVsPt"]->Fill(simPt, 1.5);
			else if(idx == 2) histContainer2D_["hCheckTracksVsPt"]->Fill(simPt, 2.5);
			else if(idx == 3) histContainer2D_["hCheckTracksVsPt"]->Fill(simPt, 3.5);
			else if(idx == 4) histContainer2D_["hCheckTracksVsPt"]->Fill(simPt, 4.5);
			else if(idx == 5) histContainer2D_["hCheckTracksVsPt"]->Fill(simPt, 5.5);
			else histContainer2D_["hCheckTracksVsPt"]->Fill(simPt, 6.5);

			if(idx == 0 && muQ*qGen < 0) histContainer2D_["hCheckChargeVsPt"]->Fill(simPt, 0.5);
			else if(idx == 1 && muQ*qGen < 0) histContainer2D_["hCheckChargeVsPt"]->Fill(simPt, 1.5);
			else if(idx == 2 && muQ*qGen < 0) histContainer2D_["hCheckChargeVsPt"]->Fill(simPt, 2.5);
			else if(idx == 3 && muQ*qGen < 0) histContainer2D_["hCheckChargeVsPt"]->Fill(simPt, 3.5);
			else if(idx == 4 && muQ*qGen < 0) histContainer2D_["hCheckChargeVsPt"]->Fill(simPt, 4.5);
			else if(idx == 5 && muQ*qGen < 0) histContainer2D_["hCheckChargeVsPt"]->Fill(simPt, 5.5);
			//else histContainer2D_["hCheckChargeVsPt"]->Fill(simPt, 6.5);

		}

	}

  }//Fine loop sulle SimTrack

}

DEFINE_FWK_MODULE(STAMuonAnalyzer);
