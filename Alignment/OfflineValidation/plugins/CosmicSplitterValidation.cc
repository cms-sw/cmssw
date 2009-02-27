// -*- C++ -*-
//
// Package:    CosmicSplitterValidation
// Class:      CosmicSplitterValidation
// 
/**\class CosmicSplitterValidation CosmicSplitterValidation.cc 
 
 Description: Takes a set of alignment constants and turns them into a ROOT file
 
 Implementation:
 <Notes on implementation>
 */
//
// Original Author:  Nhan Tran
//         Created:  Mon Jul 16m 16:56:34 CDT 2007
// $Id: CosmicSplitterValidation.cc,v 1.3 2009/01/27 13:57:09 ntran Exp $
//
//

// system include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include <algorithm>
#include "TTree.h"
#include "TFile.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventPrincipal.h" 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "TrackingTools/TransientTrack/interface/BasicTransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/Common/interface/RefToBase.h" 
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TrajectoryState/interface/CopyUsingClone.h" 
#include "RecoVertex/VertexTools/interface/PerigeeLinearizedTrackState.h" 
#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryPlugin.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"


#include <TFile.h>
#include <TH1D.h>
#include <TTree.h>
#include <TMath.h>
#include <TNtuple.h>

//
// class decleration
//
using namespace edm;

class CosmicSplitterValidation : public edm::EDAnalyzer {
public:
	explicit CosmicSplitterValidation(const edm::ParameterSet&);
	~CosmicSplitterValidation();
	
	
private:
	virtual void beginJob(const edm::EventSetup &iSetup);
	virtual void analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup);
	virtual void endJob() ;
	
	bool is_gold_muon(const edm::Event& e);
	
	edm::InputTag splitTracks_;
	edm::InputTag splitGlobalMuons_;
	edm::InputTag originalTracks_;
	edm::InputTag originalGlobalMuons_;
	
	bool checkIfGolden_;
	bool splitMuons_;
	int totalTracksToAnalyzer_;
	int goldenCtr;
	int twoTracksCtr;
	int goldenPlusTwoTracksCtr;
	int _passesTracksPlusMuonsCuts;
	edm::ESHandle<TrackerGeometry> theGeometry;
	edm::ESHandle<MagneticField>   theMagField;
	edm::ESHandle<DTGeometry>             dtGeometry;
	edm::ESHandle<CSCGeometry>            cscGeometry;
	edm::ESHandle<RPCGeometry>            rpcGeometry;
	
	
	
	edm::Service<TFileService> tfile;
	// ----------member data ---------------------------
	//std::vector<AlignTransform> m_align;
	// tree
	TTree* _splitterTree;
	// tree vars
	// split track variables
	double _dcaX1_spl, _dcaY1_spl, _dcaZ1_spl;
	double _dcaX2_spl, _dcaY2_spl, _dcaZ2_spl;
	double _dxy1_spl, _dxy2_spl, _dz1_spl, _dz2_spl;
	double _theta1_spl, _theta2_spl, _phi1_spl, _phi2_spl;
	double _ddxy_spl, _ddz_spl, _dtheta_spl, _dphi_spl;
	double _pt1_spl, _pt2_spl, _dpt_spl;
	double _eta1_spl, _eta2_spl, _deta_spl;
	// split track errors
	double _pt1Err_spl, _pt2Err_spl;
	double _theta1Err_spl, _theta2Err_spl;
	double _phi1Err_spl, _phi2Err_spl;
	double _d01Err_spl, _d02Err_spl;
	double _dz1Err_spl, _dz2Err_spl;
	// original track variables
	double _dcaX_org, _dcaY_org, _dcaZ_org;
	double _dxy_org, _dz_org;
	double _theta_org, _phi_org, _eta_org, _pt_org;
	double _norchi2_org;
	
	// split sta variables
	double _dcaX1_sta, _dcaY1_sta, _dcaZ1_sta;
	double _dcaX2_sta, _dcaY2_sta, _dcaZ2_sta;
	double _dxy1_sta, _dxy2_sta, _dz1_sta, _dz2_sta;
	double _theta1_sta, _theta2_sta, _phi1_sta, _phi2_sta;
	double _ddxy_sta, _ddz_sta, _dtheta_sta, _dphi_sta;
	double _pt1_sta, _pt2_sta, _dpt_sta;
	double _eta1_sta, _eta2_sta, _deta_sta;
	// split sta errors
	double _pt1Err_sta, _pt2Err_sta;
	double _theta1Err_sta, _theta2Err_sta;
	double _phi1Err_sta, _phi2Err_sta;
	double _d01Err_sta, _d02Err_sta;
	double _dz1Err_sta, _dz2Err_sta;

	// split glb variables
	double _dcaX1_glb, _dcaY1_glb, _dcaZ1_glb;
	double _dcaX2_glb, _dcaY2_glb, _dcaZ2_glb;
	double _dxy1_glb, _dxy2_glb, _dz1_glb, _dz2_glb;
	double _theta1_glb, _theta2_glb, _phi1_glb, _phi2_glb;
	double _ddxy_glb, _ddz_glb, _dtheta_glb, _dphi_glb;
	double _pt1_glb, _pt2_glb, _dpt_glb;
	double _eta1_glb, _eta2_glb, _deta_glb;
	double _norchi1_glb, _norchi2_glb;
	// split glb errors
	double _pt1Err_glb, _pt2Err_glb;
	double _theta1Err_glb, _theta2Err_glb;
	double _phi1Err_glb, _phi2Err_glb;
	double _d01Err_glb, _d02Err_glb;
	double _dz1Err_glb, _dz2Err_glb;
	// original glb muon variables
	double _dcaX_orm, _dcaY_orm, _dcaZ_orm;
	double _dxy_orm, _dz_orm;
	double _theta_orm, _phi_orm, _eta_orm, _pt_orm;
	double _norchi2_orm;
	
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CosmicSplitterValidation::CosmicSplitterValidation(const edm::ParameterSet& iConfig):
	splitTracks_(iConfig.getParameter<edm::InputTag>("splitTracks")),
	splitGlobalMuons_(iConfig.getParameter<edm::InputTag>("splitGlobalMuons")),
	originalTracks_(iConfig.getParameter<edm::InputTag>("originalTracks")),
	originalGlobalMuons_(iConfig.getParameter<edm::InputTag>("originalGlobalMuons")),
	checkIfGolden_(iConfig.getParameter<bool>("checkIfGolden")),
	splitMuons_(iConfig.getParameter<bool> ("ifSplitMuons"))
{
	
}


CosmicSplitterValidation::~CosmicSplitterValidation()
{}


//
// member functions
//

// ------------ method called to for each event  ------------
void CosmicSplitterValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	
	// check if golden muon
	bool isGolden = true;
	if (checkIfGolden_) isGolden = is_gold_muon( iEvent );

	// set up geometries and magnetic field
	iSetup.get<TrackerDigiGeometryRecord>().get(theGeometry);
	iSetup.get<IdealMagneticFieldRecord>().get(theMagField);
	iSetup.get<MuonGeometryRecord>().get(dtGeometry);
	iSetup.get<MuonGeometryRecord>().get(cscGeometry);
	iSetup.get<MuonGeometryRecord>().get(rpcGeometry);
	
	
	// grab collections
	edm::Handle<std::vector<reco::Track> > tracks;
	edm::Handle<reco::MuonCollection> globalMuons;
	edm::Handle<reco::MuonCollection> originalGlobalMuons;
	edm::Handle<std::vector<reco::Track> > originalTracks;
	iEvent.getByLabel(splitTracks_, tracks);
	iEvent.getByLabel(originalTracks_, originalTracks);
	if (splitMuons_){
		iEvent.getByLabel(splitGlobalMuons_, globalMuons);
		iEvent.getByLabel(originalGlobalMuons_, originalGlobalMuons);
	}
	
	totalTracksToAnalyzer_ = totalTracksToAnalyzer_ + tracks->size();
	if (isGolden) goldenCtr++;
	if (tracks->size() == 2) twoTracksCtr++;
	if (tracks->size() == 2 && originalTracks->size() == 1 && isGolden){
		goldenPlusTwoTracksCtr++;
		
		int gmCtr = 0; 
		bool topGlobalMuonFlag = false;
		bool bottomGlobalMuonFlag = false;
		int topGlobalMuon = -1;
		int bottomGlobalMuon = -1;
		double topGlobalMuonNorchi2 = 1e10;
		double bottomGlobalMuonNorchi2 = 1e10;
		
		if (splitMuons_){
			// check if split global muons are good
			for (std::vector<reco::Muon>::const_iterator gmI = globalMuons->begin(); gmI != globalMuons->end(); gmI++){
				
				if ( gmI->isTrackerMuon() && gmI->isStandAloneMuon() && gmI->isGlobalMuon() ){
					
					reco::TrackRef trackerTrackRef1( tracks, 0 );
					reco::TrackRef trackerTrackRef2( tracks, 1 );
					
					if (gmI->innerTrack() == trackerTrackRef1){
						if (gmI->globalTrack()->normalizedChi2() < topGlobalMuonNorchi2){
							topGlobalMuonFlag = true;
							topGlobalMuonNorchi2 = gmI->globalTrack()->normalizedChi2();
							topGlobalMuon = gmCtr;
						}
					}
					if (gmI->innerTrack() == trackerTrackRef2){
						if (gmI->globalTrack()->normalizedChi2() < bottomGlobalMuonNorchi2){
							bottomGlobalMuonFlag = true;
							bottomGlobalMuonNorchi2 = gmI->globalTrack()->normalizedChi2();
							bottomGlobalMuon = gmCtr;
						}
					}
				}
				gmCtr++;
			}			
		}
		
		if ( (!splitMuons_) || (splitMuons_ && bottomGlobalMuonFlag && topGlobalMuonFlag) ){
			
			_passesTracksPlusMuonsCuts++;
			
			// split tracks calculations
			reco::Track track1 = tracks->at(0);
			reco::Track track2 = tracks->at(1);
			
			math::XYZPoint dca1 = track1.referencePoint();
			math::XYZPoint dca2 = track2.referencePoint();
			
			double dtheta_Val = track1.theta() - track2.theta();
			double dphi_Val = track1.phi() - track2.phi();
			double ddxy_Val = track1.d0() - track2.d0();
			double ddz_Val = track1.dz() - track2.dz();
			double dpt_Val = track1.pt() - track2.pt();
			
			// original tracks calculations
			reco::Track origTrack = originalTracks->at(0);
			math::XYZPoint dca_org = origTrack.referencePoint();
			
			// fill tree
			// split tracks
			_dcaX1_spl = dca1.x(); 
			_dcaY1_spl = dca1.y();
			_dcaZ1_spl = dca1.z();
			_dcaX2_spl = dca2.x(); 
			_dcaY2_spl = dca2.y();
			_dcaZ2_spl = dca2.z();
			_dxy1_spl = track1.d0();
			_dxy2_spl = track2.d0();
			_dz1_spl = track1.dz();
			_dz2_spl = track2.dz();
			_d01Err_spl = track1.d0Error();
			_d02Err_spl = track2.d0Error();
			_dz1Err_spl = track1.dzError();
			_dz2Err_spl = track2.dzError();
			_theta1_spl = track1.theta();
			_theta2_spl = track2.theta();
			_theta1Err_spl = track1.thetaError();
			_theta2Err_spl = track2.thetaError();
			_phi1_spl = track1.phi();
			_phi2_spl = track2.phi();
			_phi1Err_spl = track1.phiError();
			_phi2Err_spl = track2.phiError();
			_ddxy_spl = ddxy_Val;
			_ddz_spl = ddz_Val;
			_dtheta_spl = dtheta_Val;
			_dphi_spl = dphi_Val;
			_pt1_spl = track1.pt();
			_pt2_spl = track2.pt();
			_pt1Err_spl = track1.ptError();
			_pt2Err_spl = track2.ptError();
			_dpt_spl = dpt_Val;
			_eta1_spl = track1.eta();
			_eta2_spl = track2.eta();
			_deta_spl = _eta1_spl - _eta2_spl;
			
			// original tracks
			_dcaX_org = dca_org.x();
			_dcaY_org = dca_org.y();
			_dcaZ_org = dca_org.z();
			_dxy_org = origTrack.d0();
			_dz_org = origTrack.dz();
			_theta_org = origTrack.theta();
			_phi_org = origTrack.phi();
			_eta_org = origTrack.eta();
			_pt_org = origTrack.pt();
			_norchi2_org = origTrack.normalizedChi2();
			
			// split muons calculations
			if (splitMuons_){
				
				reco::Muon muonTop = globalMuons->at( topGlobalMuon );
				reco::Muon muonBottom = globalMuons->at( bottomGlobalMuon );				
				
				reco::TrackRef glb1 = muonTop.globalTrack();
				reco::TrackRef glb2 = muonBottom.globalTrack();
				reco::TrackRef sta1 = muonTop.outerTrack();
				reco::TrackRef sta2 = muonBottom.outerTrack();
				
				// standalone muon variables
				_dcaX1_sta = sta1->referencePoint().x();
				_dcaY1_sta = sta1->referencePoint().y();
				_dcaZ1_sta = sta1->referencePoint().z();
				_dcaX2_sta = sta2->referencePoint().x();
				_dcaY2_sta = sta2->referencePoint().y();
				_dcaZ2_sta = sta2->referencePoint().z();
				_dxy1_sta = sta1->d0();
				_dxy2_sta = sta2->d0();
				_dz1_sta = sta1->dz();
				_dz2_sta = sta2->dz();
				_d01Err_sta = sta1->d0Error();
				_d02Err_sta = sta2->d0Error();
				_dz1Err_sta = sta1->dzError();
				_dz2Err_sta = sta2->dzError();
				_theta1_sta = sta1->theta();
				_theta2_sta = sta2->theta();
				_theta1Err_sta = sta1->thetaError();
				_theta2Err_sta = sta2->thetaError();
				_phi1_sta = sta1->phi();
				_phi2_sta = sta2->phi();
				_phi1Err_sta = sta1->phiError();
				_phi2Err_sta = sta2->phiError();
				_ddxy_sta = sta1->d0() - sta2->d0();
				_ddz_sta = sta1->dz() - sta2->dz();
				_dtheta_sta = sta1->theta() - sta2->theta();
				_dphi_sta = sta1->phi() - sta2->phi();
				_pt1_sta = sta1->pt();
				_pt2_sta = sta2->pt();
				_pt1Err_sta = sta1->ptError();
				_pt2Err_sta = sta2->ptError();
				_dpt_sta = sta1->pt() - sta2->pt();
				_eta1_sta = sta1->eta();
				_eta2_sta = sta2->eta();
				_deta_sta = _eta1_sta - _eta2_sta;
				
				// global muon variables
				_dcaX1_glb = glb1->referencePoint().x();
				_dcaY1_glb = glb1->referencePoint().y();
				_dcaZ1_glb = glb1->referencePoint().z();
				_dcaX2_glb = glb2->referencePoint().x();
				_dcaY2_glb = glb2->referencePoint().y();
				_dcaZ2_glb = glb2->referencePoint().z();
				_dxy1_glb = glb1->d0();
				_dxy2_glb = glb2->d0();
				_dz1_glb = glb1->dz();
				_dz2_glb = glb2->dz();
				_d01Err_glb = glb1->d0Error();
				_d02Err_glb = glb2->d0Error();
				_dz1Err_glb = glb1->dzError();
				_dz2Err_glb = glb2->dzError();
				_theta1_glb = glb1->theta();
				_theta2_glb = glb2->theta();
				_theta1Err_glb = glb1->thetaError();
				_theta2Err_glb = glb2->thetaError();
				_phi1_glb = glb1->phi();
				_phi2_glb = glb2->phi();
				_phi1Err_glb = glb1->phiError();
				_phi2Err_glb = glb2->phiError();
				_ddxy_glb = glb1->d0() - glb2->d0();
				_ddz_glb = glb1->dz() - glb2->dz();
				_dtheta_glb = glb1->theta() - glb2->theta();
				_dphi_glb = glb1->phi() - glb2->phi();
				_pt1_glb = glb1->pt();
				_pt2_glb = glb2->pt();
				_pt1Err_glb = glb1->ptError();
				_pt2Err_glb = glb2->ptError();
				_dpt_glb = glb1->pt() - glb2->pt();
				_eta1_glb = glb1->eta();
				_eta2_glb = glb2->eta();
				_deta_glb = _eta1_glb - _eta2_glb;
				_norchi1_glb = glb1->normalizedChi2();
				_norchi2_glb = glb2->normalizedChi2();
				
			}
			
			
			_splitterTree->Fill();
		}
	}
	
	
}


// ------------ method called once each job just before starting event loop  ------------
void CosmicSplitterValidation::beginJob(const edm::EventSetup& iSetup)
{
	edm::LogInfo("beginJob") << "Begin Job" << std::endl;
	
	_splitterTree = tfile->make<TTree>("splitterTree","splitterTree");
	
	_splitterTree->Branch("dcaX1_spl", &_dcaX1_spl, "dcaX1_spl/D");
	_splitterTree->Branch("dcaY1_spl", &_dcaY1_spl, "dcaY1_spl/D");
	_splitterTree->Branch("dcaZ1_spl", &_dcaZ1_spl, "dcaZ1_spl/D");
	_splitterTree->Branch("dcaX2_spl", &_dcaX2_spl, "dcaX2_spl/D");
	_splitterTree->Branch("dcaY2_spl", &_dcaY2_spl, "dcaY2_spl/D");
	_splitterTree->Branch("dcaZ2_spl", &_dcaZ2_spl, "dcaZ2_spl/D");
	_splitterTree->Branch("dxy1_spl", &_dxy1_spl, "dxy1_spl/D");
	_splitterTree->Branch("dz1_spl", &_dz1_spl, "dz1_spl/D");
	_splitterTree->Branch("dxy2_spl", &_dxy2_spl, "dxy2_spl/D");
	_splitterTree->Branch("dz2_spl", &_dz2_spl, "dz2_spl/D");
	_splitterTree->Branch("theta1_spl", &_theta1_spl, "theta1_spl/D");
	_splitterTree->Branch("theta2_spl", &_theta2_spl, "theta2_spl/D");
	_splitterTree->Branch("phi1_spl", &_phi1_spl, "phi1_spl/D");
	_splitterTree->Branch("phi2_spl", &_phi2_spl, "phi2_spl/D");
	_splitterTree->Branch("ddxy_spl", &_ddxy_spl, "ddxy_spl/D");
	_splitterTree->Branch("ddz_spl", &_ddz_spl, "ddz_spl/D");
	_splitterTree->Branch("dphi_spl", &_dphi_spl, "dphi_spl/D");
	_splitterTree->Branch("dtheta_spl", &_dtheta_spl, "dtheta_spl/D");
	_splitterTree->Branch("pt1_spl", &_pt1_spl, "pt1_spl/D");
	_splitterTree->Branch("pt2_spl", &_pt2_spl, "pt2_spl/D");
	_splitterTree->Branch("dpt_spl", &_dpt_spl, "dpt_spl/D");
	_splitterTree->Branch("eta1_spl", &_eta1_spl, "eta1_spl/D");
	_splitterTree->Branch("eta2_spl", &_eta2_spl, "eta2_spl/D");
	_splitterTree->Branch("deta_spl", &_deta_spl, "deta_spl/D");
	
	_splitterTree->Branch("d01Err_spl", &_d01Err_spl, "d01Err_spl/D");
	_splitterTree->Branch("d02Err_spl", &_d02Err_spl, "d02Err_spl/D");
	_splitterTree->Branch("dz1Err_spl", &_dz1Err_spl, "dz1Err_spl/D");
	_splitterTree->Branch("dz2Err_spl", &_dz2Err_spl, "dz2Err_spl/D");
	_splitterTree->Branch("phi1Err_spl", &_phi1Err_spl, "phi1Err_spl/D");
	_splitterTree->Branch("phi2Err_spl", &_phi2Err_spl, "phi2Err_spl/D");
	_splitterTree->Branch("theta1Err_spl", &_theta1Err_spl, "theta1Err_spl/D");
	_splitterTree->Branch("theta2Err_spl", &_theta2Err_spl, "theta2Err_spl/D");
	_splitterTree->Branch("pt1Err_spl", &_pt1Err_spl, "pt1Err_spl/D");
	_splitterTree->Branch("pt2Err_spl", &_pt2Err_spl, "pt2Err_spl/D");
	
	_splitterTree->Branch("dcaX_org", &_dcaX_org, "dcaX_org/D");
	_splitterTree->Branch("dcaY_org", &_dcaY_org, "dcaY_org/D");
	_splitterTree->Branch("dcaZ_org", &_dcaZ_org, "dcaZ_org/D");
	_splitterTree->Branch("dxy_org", &_dxy_org, "dxy_org/D");
	_splitterTree->Branch("dz_org", &_dz_org, "dz_org/D");
	_splitterTree->Branch("theta_org", &_theta_org, "theta_org/D");
	_splitterTree->Branch("phi_org", &_phi_org, "phi_org/D");
	_splitterTree->Branch("eta_org", &_eta_org, "eta_org/D");
	_splitterTree->Branch("pt_org", &_pt_org, "pt_org/D");
	_splitterTree->Branch("norchi2_org", &_norchi2_org, "norchi2_org/D");
	
	if (splitMuons_){
		
		// standalone split 
		_splitterTree->Branch("dcaX1_sta", &_dcaX1_sta, "dcaX1_sta/D");
		_splitterTree->Branch("dcaY1_sta", &_dcaY1_sta, "dcaY1_sta/D");
		_splitterTree->Branch("dcaZ1_sta", &_dcaZ1_sta, "dcaZ1_sta/D");
		_splitterTree->Branch("dcaX2_sta", &_dcaX2_sta, "dcaX2_sta/D");
		_splitterTree->Branch("dcaY2_sta", &_dcaY2_sta, "dcaY2_sta/D");
		_splitterTree->Branch("dcaZ2_sta", &_dcaZ2_sta, "dcaZ2_sta/D");
		_splitterTree->Branch("dxy1_sta", &_dxy1_sta, "dxy1_sta/D");
		_splitterTree->Branch("dz1_sta", &_dz1_sta, "dz1_sta/D");
		_splitterTree->Branch("dxy2_sta", &_dxy2_sta, "dxy2_sta/D");
		_splitterTree->Branch("dz2_sta", &_dz2_sta, "dz2_sta/D");		
		_splitterTree->Branch("theta1_sta", &_theta1_sta, "theta1_sta/D");
		_splitterTree->Branch("theta2_sta", &_theta2_sta, "theta2_sta/D");
		_splitterTree->Branch("phi1_sta", &_phi1_sta, "phi1_sta/D");
		_splitterTree->Branch("phi2_sta", &_phi2_sta, "phi2_sta/D");
		_splitterTree->Branch("ddxy_sta", &_ddxy_sta, "ddxy_sta/D");
		_splitterTree->Branch("ddz_sta", &_ddz_sta, "ddz_sta/D");
		_splitterTree->Branch("dphi_sta", &_dphi_sta, "dphi_sta/D");
		_splitterTree->Branch("dtheta_sta", &_dtheta_sta, "dtheta_sta/D");
		_splitterTree->Branch("pt1_sta", &_pt1_sta, "pt1_sta/D");
		_splitterTree->Branch("pt2_sta", &_pt2_sta, "pt2_sta/D");
		_splitterTree->Branch("dpt_sta", &_dpt_sta, "dpt_sta/D");
		_splitterTree->Branch("eta1_sta", &_eta1_sta, "eta1_sta/D");
		_splitterTree->Branch("eta2_sta", &_eta2_sta, "eta2_sta/D");
		_splitterTree->Branch("deta_sta", &_deta_sta, "deta_sta/D");
		
		_splitterTree->Branch("d01Err_sta", &_d01Err_sta, "d01Err_sta/D");
		_splitterTree->Branch("d02Err_sta", &_d02Err_sta, "d02Err_sta/D");
		_splitterTree->Branch("dz1Err_sta", &_dz1Err_sta, "dz1Err_sta/D");
		_splitterTree->Branch("dz2Err_sta", &_dz2Err_sta, "dz2Err_sta/D");
		_splitterTree->Branch("phi1Err_sta", &_phi1Err_sta, "phi1Err_sta/D");
		_splitterTree->Branch("phi2Err_sta", &_phi2Err_sta, "phi2Err_sta/D");
		_splitterTree->Branch("theta1Err_sta", &_theta1Err_sta, "theta1Err_sta/D");
		_splitterTree->Branch("theta2Err_sta", &_theta2Err_sta, "theta2Err_sta/D");
		_splitterTree->Branch("pt1Err_sta", &_pt1Err_sta, "pt1Err_sta/D");
		_splitterTree->Branch("pt2Err_sta", &_pt2Err_sta, "pt2Err_sta/D");
		
		// global split 
		_splitterTree->Branch("dcaX1_glb", &_dcaX1_glb, "dcaX1_glb/D");
		_splitterTree->Branch("dcaY1_glb", &_dcaY1_glb, "dcaY1_glb/D");
		_splitterTree->Branch("dcaZ1_glb", &_dcaZ1_glb, "dcaZ1_glb/D");
		_splitterTree->Branch("dcaX2_glb", &_dcaX2_glb, "dcaX2_glb/D");
		_splitterTree->Branch("dcaY2_glb", &_dcaY2_glb, "dcaY2_glb/D");
		_splitterTree->Branch("dcaZ2_glb", &_dcaZ2_glb, "dcaZ2_glb/D");
		_splitterTree->Branch("dxy1_glb", &_dxy1_glb, "dxy1_glb/D");
		_splitterTree->Branch("dz1_glb", &_dz1_glb, "dz1_glb/D");
		_splitterTree->Branch("dxy2_glb", &_dxy2_glb, "dxy2_glb/D");
		_splitterTree->Branch("dz2_glb", &_dz2_glb, "dz2_glb/D");		
		_splitterTree->Branch("theta1_glb", &_theta1_glb, "theta1_glb/D");
		_splitterTree->Branch("theta2_glb", &_theta2_glb, "theta2_glb/D");
		_splitterTree->Branch("phi1_glb", &_phi1_glb, "phi1_glb/D");
		_splitterTree->Branch("phi2_glb", &_phi2_glb, "phi2_glb/D");
		_splitterTree->Branch("ddxy_glb", &_ddxy_glb, "ddxy_glb/D");
		_splitterTree->Branch("ddz_glb", &_ddz_glb, "ddz_glb/D");
		_splitterTree->Branch("dphi_glb", &_dphi_glb, "dphi_glb/D");
		_splitterTree->Branch("dtheta_glb", &_dtheta_glb, "dtheta_glb/D");
		_splitterTree->Branch("pt1_glb", &_pt1_glb, "pt1_glb/D");
		_splitterTree->Branch("pt2_glb", &_pt2_glb, "pt2_glb/D");
		_splitterTree->Branch("dpt_glb", &_dpt_glb, "dpt_glb/D");
		_splitterTree->Branch("eta1_glb", &_eta1_glb, "eta1_glb/D");
		_splitterTree->Branch("eta2_glb", &_eta2_glb, "eta2_glb/D");
		_splitterTree->Branch("deta_glb", &_deta_glb, "deta_glb/D");
		_splitterTree->Branch("norchi1_glb", &_norchi1_glb, "norchi1_glb/D");
		_splitterTree->Branch("norchi2_glb", &_norchi2_glb, "norchi2_glb/D");

		_splitterTree->Branch("d01Err_glb", &_d01Err_glb, "d01Err_glb/D");
		_splitterTree->Branch("d02Err_glb", &_d02Err_glb, "d02Err_glb/D");
		_splitterTree->Branch("dz1Err_glb", &_dz1Err_glb, "dz1Err_glb/D");
		_splitterTree->Branch("dz2Err_glb", &_dz2Err_glb, "dz2Err_glb/D");
		_splitterTree->Branch("phi1Err_glb", &_phi1Err_glb, "phi1Err_glb/D");
		_splitterTree->Branch("phi2Err_glb", &_phi2Err_glb, "phi2Err_glb/D");
		_splitterTree->Branch("theta1Err_glb", &_theta1Err_glb, "theta1Err_glb/D");
		_splitterTree->Branch("theta2Err_glb", &_theta2Err_glb, "theta2Err_glb/D");
		_splitterTree->Branch("pt1Err_glb", &_pt1Err_glb, "pt1Err_glb/D");
		_splitterTree->Branch("pt2Err_glb", &_pt2Err_glb, "pt2Err_glb/D");

	}
	
	totalTracksToAnalyzer_ = 0;
	goldenCtr = 0;
	twoTracksCtr = 0;
	goldenPlusTwoTracksCtr = 0;
	_passesTracksPlusMuonsCuts = 0;
}


// ------------ method called once each job just after ending the event loop  ------------
void CosmicSplitterValidation::endJob() {
	
	//std::cout << "totalTracksToAnalyzer: " << totalTracksToAnalyzer_ << std::endl;
	std::cout << "golden: " << goldenCtr << ", two tracks: " << twoTracksCtr << ", golden+twotracks: " << goldenPlusTwoTracksCtr << ", tracks+muons cuts: " << _passesTracksPlusMuonsCuts << std::endl;
}

bool CosmicSplitterValidation::is_gold_muon(const edm::Event& e){
	
	edm::Handle<reco::MuonCollection> muHandle;
	e.getByLabel("STAMuons", muHandle);
	const reco::MuonCollection & muons = *(muHandle.product());
	// make sure there are 2 muons
	if ( 2 != muons.size() ) return false;
	
	double mudd0=0., mudphi=0., muddsz=0., mudeta=0.;
	for ( unsigned int bindex = 0; bindex < muons.size(); ++bindex ) {
		reco::Muon mymuon = muons[bindex];
		// deprecated in 21x (now outerTrack)
		//reco::TrackRef mutrackref = mymuon.standAloneMuon();
		reco::TrackRef mutrackref = mymuon.outerTrack();
		const reco::Track* mutrack = mutrackref.get();
		if (0 == bindex){
			mudd0+=mutrack->d0(); 
			mudphi+=mutrack->phi();
			muddsz+=mutrack->dsz(); 
			mudeta+=mymuon.eta();
		}
		if (1 == bindex){
			mudd0-=mutrack->d0(); 
			mudphi-=mutrack->phi();
			muddsz-=mutrack->dsz(); 
			mudeta-=mymuon.eta();
		}
	}
	if ((fabs(mudd0)<15.0)&&(fabs(mudphi)<0.045)&&(fabs(muddsz)<20.0)&&(fabs(mudeta)<0.060)) return true;
	return false;
}


//define this as a plug-in
DEFINE_FWK_MODULE(CosmicSplitterValidation);
