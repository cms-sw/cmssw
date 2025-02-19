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
// $Id: CosmicSplitterValidation.cc,v 1.12 2011/12/20 15:11:41 mussgill Exp $
//
//

// system include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <algorithm>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
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

#include <TTree.h>

//
// class decleration
//

class CosmicSplitterValidation : public edm::EDAnalyzer {
public:
	explicit CosmicSplitterValidation(const edm::ParameterSet&);
	~CosmicSplitterValidation();
	
	
private:
	virtual void beginJob();
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
	
	edm::Service<TFileService> tfile;
	// ----------member data ---------------------------
	//std::vector<AlignTransform> m_align;
	// tree
	TTree* splitterTree_;
	// tree vars
	// split track variables
	double dcaX1_spl_, dcaY1_spl_, dcaZ1_spl_;
	double dcaX2_spl_, dcaY2_spl_, dcaZ2_spl_;
	double dxy1_spl_, dxy2_spl_, dz1_spl_, dz2_spl_;
	double theta1_spl_, theta2_spl_, phi1_spl_, phi2_spl_;
	double ddxy_spl_, ddz_spl_, dtheta_spl_, dphi_spl_;
	double pt1_spl_, pt2_spl_, dpt_spl_, p1_spl_, p2_spl_;
	double eta1_spl_, eta2_spl_, deta_spl_;
        double nHits1_spl_, nHitsPXB1_spl_, nHitsPXF1_spl_, nHitsTIB1_spl_;
        double nHitsTOB1_spl_, nHitsTID1_spl_, nHitsTEC1_spl_;
        double nHits2_spl_, nHitsPXB2_spl_, nHitsPXF2_spl_, nHitsTIB2_spl_;
        double nHitsTOB2_spl_, nHitsTID2_spl_, nHitsTEC2_spl_;
	// spl_it track errors
	double pt1Err_spl_, pt2Err_spl_;
	double theta1Err_spl_, theta2Err_spl_;
	double phi1Err_spl_, phi2Err_spl_;
	double d01Err_spl_, d02Err_spl_;
	double dz1Err_spl_, dz2Err_spl_;
	// original track variables
	double dcaX_org_, dcaY_org_, dcaZ_org_;
	double dxy_org_, dz_org_;
	double theta_org_, phi_org_, eta_org_, pt_org_, p_org_;
	double norchi2_org_;
	
	// split sta variables
	double dcaX1_sta_, dcaY1_sta_, dcaZ1_sta_;
	double dcaX2_sta_, dcaY2_sta_, dcaZ2_sta_;
	double dxy1_sta_, dxy2_sta_, dz1_sta_, dz2_sta_;
	double theta1_sta_, theta2_sta_, phi1_sta_, phi2_sta_;
	double ddxy_sta_, ddz_sta_, dtheta_sta_, dphi_sta_;
	double pt1_sta_, pt2_sta_, dpt_sta_, p1_sta_, p2_sta_;
	double eta1_sta_, eta2_sta_, deta_sta_;
	// split sta_ errors
	double pt1Err_sta_, pt2Err_sta_;
	double theta1Err_sta_, theta2Err_sta_;
	double phi1Err_sta_, phi2Err_sta_;
	double d01Err_sta_, d02Err_sta_;
	double dz1Err_sta_, dz2Err_sta_;

	// split glb_ variables
	double dcaX1_glb_, dcaY1_glb_, dcaZ1_glb_;
	double dcaX2_glb_, dcaY2_glb_, dcaZ2_glb_;
	double dxy1_glb_, dxy2_glb_, dz1_glb_, dz2_glb_;
	double theta1_glb_, theta2_glb_, phi1_glb_, phi2_glb_;
	double ddxy_glb_, ddz_glb_, dtheta_glb_, dphi_glb_;
	double pt1_glb_, pt2_glb_, dpt_glb_, p1_glb_, p2_glb_;
	double eta1_glb_, eta2_glb_, deta_glb_;
	double norchi1_glb_, norchi2_glb_;
	// split glb_ errors
	double pt1Err_glb_, pt2Err_glb_;
	double theta1Err_glb_, theta2Err_glb_;
	double phi1Err_glb_, phi2Err_glb_;
	double d01Err_glb_, d02Err_glb_;
	double dz1Err_glb_, dz2Err_glb_;
	// original glb muon variables
	double dcaX_orm_, dcaY_orm_, dcaZ_orm_;
	double dxy_orm_, dz_orm_;
	double theta_orm_, phi_orm_, eta_orm_, pt_orm_, p_orm_;
	double norchi2_orm_;
	
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
	splitMuons_(iConfig.getParameter<bool> ("ifSplitMuons")),
	totalTracksToAnalyzer_(0),
	goldenCtr(0),
	twoTracksCtr(0),
	goldenPlusTwoTracksCtr(0),
	_passesTracksPlusMuonsCuts(0),
	splitterTree_(0),
	dcaX1_spl_(0), dcaY1_spl_(0), dcaZ1_spl_(0),
	dcaX2_spl_(0), dcaY2_spl_(0), dcaZ2_spl_(0),
	dxy1_spl_(0), dxy2_spl_(0), dz1_spl_(0), dz2_spl_(0),
	theta1_spl_(0), theta2_spl_(0), phi1_spl_(0), phi2_spl_(0),
	ddxy_spl_(0), ddz_spl_(0), dtheta_spl_(0), dphi_spl_(0),
	pt1_spl_(0), pt2_spl_(0), dpt_spl_(0), p1_spl_(0), p2_spl_(0),
	eta1_spl_(0), eta2_spl_(0), deta_spl_(0),
	nHits1_spl_(0), nHitsPXB1_spl_(0), nHitsPXF1_spl_(0), nHitsTIB1_spl_(0),
	nHitsTOB1_spl_(0), nHitsTID1_spl_(0), nHitsTEC1_spl_(0),
	nHits2_spl_(0), nHitsPXB2_spl_(0), nHitsPXF2_spl_(0), nHitsTIB2_spl_(0),
	nHitsTOB2_spl_(0), nHitsTID2_spl_(0), nHitsTEC2_spl_(0),
	pt1Err_spl_(0), pt2Err_spl_(0),
	theta1Err_spl_(0), theta2Err_spl_(0),
	phi1Err_spl_(0), phi2Err_spl_(0),
	d01Err_spl_(0), d02Err_spl_(0),
	dz1Err_spl_(0), dz2Err_spl_(0),
	dcaX_org_(0), dcaY_org_(0), dcaZ_org_(0),
	dxy_org_(0), dz_org_(0),
	theta_org_(0), phi_org_(0), eta_org_(0), pt_org_(0), p_org_(0),
	norchi2_org_(0),
	dcaX1_sta_(0), dcaY1_sta_(0), dcaZ1_sta_(0),
	dcaX2_sta_(0), dcaY2_sta_(0), dcaZ2_sta_(0),
	dxy1_sta_(0), dxy2_sta_(0), dz1_sta_(0), dz2_sta_(0),
	theta1_sta_(0), theta2_sta_(0), phi1_sta_(0), phi2_sta_(0),
	ddxy_sta_(0), ddz_sta_(0), dtheta_sta_(0), dphi_sta_(0),
	pt1_sta_(0), pt2_sta_(0), dpt_sta_(0), p1_sta_(0), p2_sta_(0),
	eta1_sta_(0), eta2_sta_(0), deta_sta_(0),
	pt1Err_sta_(0), pt2Err_sta_(0),
	theta1Err_sta_(0), theta2Err_sta_(0),
	phi1Err_sta_(0), phi2Err_sta_(0),
	d01Err_sta_(0), d02Err_sta_(0),
	dz1Err_sta_(0), dz2Err_sta_(0),
	dcaX1_glb_(0), dcaY1_glb_(0), dcaZ1_glb_(0),
	dcaX2_glb_(0), dcaY2_glb_(0), dcaZ2_glb_(0),
	dxy1_glb_(0), dxy2_glb_(0), dz1_glb_(0), dz2_glb_(0),
	theta1_glb_(0), theta2_glb_(0), phi1_glb_(0), phi2_glb_(0),
	ddxy_glb_(0), ddz_glb_(0), dtheta_glb_(0), dphi_glb_(0),
	pt1_glb_(0), pt2_glb_(0), dpt_glb_(0), p1_glb_(0), p2_glb_(0),
	eta1_glb_(0), eta2_glb_(0), deta_glb_(0),
	norchi1_glb_(0), norchi2_glb_(0),
	pt1Err_glb_(0), pt2Err_glb_(0),
	theta1Err_glb_(0), theta2Err_glb_(0),
	phi1Err_glb_(0), phi2Err_glb_(0),
	d01Err_glb_(0), d02Err_glb_(0),
	dz1Err_glb_(0), dz2Err_glb_(0),
	dcaX_orm_(0), dcaY_orm_(0), dcaZ_orm_(0),
	dxy_orm_(0), dz_orm_(0),
	theta_orm_(0), phi_orm_(0), eta_orm_(0), pt_orm_(0), p_orm_(0),
	norchi2_orm_(0)
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

	const int kBPIX = PixelSubdetector::PixelBarrel;
	const int kFPIX = PixelSubdetector::PixelEndcap;

	
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
			
			// looping through the hits for track 1
			double Nrechits1 =0;
			double nhitinTIB1 =0; 
			double nhitinTOB1 =0; 
			double nhitinTID1 =0; 
			double nhitinTEC1 =0; 
			double nhitinBPIX1 =0;
			double nhitinFPIX1 =0;    
			for (trackingRecHit_iterator iHit = track1.recHitsBegin(); iHit != track1.recHitsEnd(); ++iHit) {
				
				if((*iHit)->isValid()) {       
					
					Nrechits1++;
					
					int type =(*iHit)->geographicalId().subdetId();
					
					if(type==int(StripSubdetector::TIB)){++nhitinTIB1;}
					if(type==int(StripSubdetector::TOB)){++nhitinTOB1;}
					if(type==int(StripSubdetector::TID)){++nhitinTID1;}
					if(type==int(StripSubdetector::TEC)){++nhitinTEC1;}
					if(type==int(                kBPIX)){++nhitinBPIX1;}
					if(type==int(                kFPIX)){++nhitinFPIX1;}
					
				}
			}    
			nHits1_spl_ = Nrechits1;
			nHitsTIB1_spl_ = nhitinTIB1; 
			nHitsTOB1_spl_ = nhitinTOB1; 
			nHitsTID1_spl_ = nhitinTID1; 
			nHitsTEC1_spl_ = nhitinTEC1; 
			nHitsPXB1_spl_ = nhitinBPIX1;
			nHitsPXF1_spl_ = nhitinFPIX1;  
			
			// looping through the hits for track 2
			double Nrechits2 =0;
			double nhitinTIB2 =0; 
			double nhitinTOB2 =0; 
			double nhitinTID2 =0; 
			double nhitinTEC2 =0; 
			double nhitinBPIX2 =0;
			double nhitinFPIX2 =0;    
			for (trackingRecHit_iterator iHit = track2.recHitsBegin(); iHit != track2.recHitsEnd(); ++iHit) {
				
				if((*iHit)->isValid()) {       
					
					Nrechits2++;
					
					int type =(*iHit)->geographicalId().subdetId();
					
					if(type==int(StripSubdetector::TIB)){++nhitinTIB2;}
					if(type==int(StripSubdetector::TOB)){++nhitinTOB2;}
					if(type==int(StripSubdetector::TID)){++nhitinTID2;}
					//\\if(type==int(StripSubdetector::TEC)){++nhitinTEC2;}
					if(type==int(                kBPIX)){++nhitinBPIX2;}
					if(type==int(                kFPIX)){++nhitinFPIX2;}
					
				}
			}    
			nHits2_spl_ = Nrechits2;
			nHitsTIB2_spl_ = nhitinTIB2; 
			nHitsTOB2_spl_ = nhitinTOB2; 
			nHitsTID2_spl_ = nhitinTID2; 
			nHitsTEC2_spl_ = nhitinTEC2; 
			nHitsPXB2_spl_ = nhitinBPIX2;
			nHitsPXF2_spl_ = nhitinFPIX2;    
			
			
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
			dcaX1_spl_ = dca1.x(); 
			dcaY1_spl_ = dca1.y();
			dcaZ1_spl_ = dca1.z();
			dcaX2_spl_ = dca2.x(); 
			dcaY2_spl_ = dca2.y();
			dcaZ2_spl_ = dca2.z();
			dxy1_spl_ = track1.d0();
			dxy2_spl_ = track2.d0();
			dz1_spl_ = track1.dz();
			dz2_spl_ = track2.dz();
			d01Err_spl_ = track1.d0Error();
			d02Err_spl_ = track2.d0Error();
			dz1Err_spl_ = track1.dzError();
			dz2Err_spl_ = track2.dzError();
			theta1_spl_ = track1.theta();
			theta2_spl_ = track2.theta();
			theta1Err_spl_ = track1.thetaError();
			theta2Err_spl_ = track2.thetaError();
			phi1_spl_ = track1.phi();
			phi2_spl_ = track2.phi();
			phi1Err_spl_ = track1.phiError();
			phi2Err_spl_ = track2.phiError();
			ddxy_spl_ = ddxy_Val;
			ddz_spl_ = ddz_Val;
			dtheta_spl_ = dtheta_Val;
			dphi_spl_ = dphi_Val;
			pt1_spl_ = track1.pt();
			pt2_spl_ = track2.pt();
			p1_spl_ = track1.p();
			p2_spl_ = track2.p();
			pt1Err_spl_ = track1.ptError();
			pt2Err_spl_ = track2.ptError();
			dpt_spl_ = dpt_Val;
			eta1_spl_ = track1.eta();
			eta2_spl_ = track2.eta();
			deta_spl_ = eta1_spl_ - eta2_spl_;
			
			// original tracks
			dcaX_org_ = dca_org.x();
			dcaY_org_ = dca_org.y();
			dcaZ_org_ = dca_org.z();
			dxy_org_ = origTrack.d0();
			dz_org_ = origTrack.dz();
			theta_org_ = origTrack.theta();
			phi_org_ = origTrack.phi();
			eta_org_ = origTrack.eta();
			pt_org_ = origTrack.pt();
			p_org_ = origTrack.p();
			norchi2_org_ = origTrack.normalizedChi2();
			
			// split muons calculations
			if (splitMuons_){
				
				reco::Muon muonTop = globalMuons->at( topGlobalMuon );
				reco::Muon muonBottom = globalMuons->at( bottomGlobalMuon );				
				
				reco::TrackRef glb1 = muonTop.globalTrack();
				reco::TrackRef glb2 = muonBottom.globalTrack();
				reco::TrackRef sta1 = muonTop.outerTrack();
				reco::TrackRef sta2 = muonBottom.outerTrack();
				
				// standalone muon variables
				dcaX1_sta_ = sta1->referencePoint().x();
				dcaY1_sta_ = sta1->referencePoint().y();
				dcaZ1_sta_ = sta1->referencePoint().z();
				dcaX2_sta_ = sta2->referencePoint().x();
				dcaY2_sta_ = sta2->referencePoint().y();
				dcaZ2_sta_ = sta2->referencePoint().z();
				dxy1_sta_ = sta1->d0();
				dxy2_sta_ = sta2->d0();
				dz1_sta_ = sta1->dz();
				dz2_sta_ = sta2->dz();
				d01Err_sta_ = sta1->d0Error();
				d02Err_sta_ = sta2->d0Error();
				dz1Err_sta_ = sta1->dzError();
				dz2Err_sta_ = sta2->dzError();
				theta1_sta_ = sta1->theta();
				theta2_sta_ = sta2->theta();
				theta1Err_sta_ = sta1->thetaError();
				theta2Err_sta_ = sta2->thetaError();
				phi1_sta_ = sta1->phi();
				phi2_sta_ = sta2->phi();
				phi1Err_sta_ = sta1->phiError();
				phi2Err_sta_ = sta2->phiError();
				ddxy_sta_ = sta1->d0() - sta2->d0();
				ddz_sta_ = sta1->dz() - sta2->dz();
				dtheta_sta_ = sta1->theta() - sta2->theta();
				dphi_sta_ = sta1->phi() - sta2->phi();
				pt1_sta_ = sta1->pt();
				pt2_sta_ = sta2->pt();
				p1_sta_ = sta1->p();
				p2_sta_ = sta2->p();
				pt1Err_sta_ = sta1->ptError();
				pt2Err_sta_ = sta2->ptError();
				dpt_sta_ = sta1->pt() - sta2->pt();
				eta1_sta_ = sta1->eta();
				eta2_sta_ = sta2->eta();
				deta_sta_ = eta1_sta_ - eta2_sta_;
				
				// global muon variables
				dcaX1_glb_ = glb1->referencePoint().x();
				dcaY1_glb_ = glb1->referencePoint().y();
				dcaZ1_glb_ = glb1->referencePoint().z();
				dcaX2_glb_ = glb2->referencePoint().x();
				dcaY2_glb_ = glb2->referencePoint().y();
				dcaZ2_glb_ = glb2->referencePoint().z();
				dxy1_glb_ = glb1->d0();
				dxy2_glb_ = glb2->d0();
				dz1_glb_ = glb1->dz();
				dz2_glb_ = glb2->dz();
				d01Err_glb_ = glb1->d0Error();
				d02Err_glb_ = glb2->d0Error();
				dz1Err_glb_ = glb1->dzError();
				dz2Err_glb_ = glb2->dzError();
				theta1_glb_ = glb1->theta();
				theta2_glb_ = glb2->theta();
				theta1Err_glb_ = glb1->thetaError();
				theta2Err_glb_ = glb2->thetaError();
				phi1_glb_ = glb1->phi();
				phi2_glb_ = glb2->phi();
				phi1Err_glb_ = glb1->phiError();
				phi2Err_glb_ = glb2->phiError();
				ddxy_glb_ = glb1->d0() - glb2->d0();
				ddz_glb_ = glb1->dz() - glb2->dz();
				dtheta_glb_ = glb1->theta() - glb2->theta();
				dphi_glb_ = glb1->phi() - glb2->phi();
				pt1_glb_ = glb1->pt();
				pt2_glb_ = glb2->pt();
				p1_glb_ = glb1->p();
				p2_glb_ = glb2->p();
				pt1Err_glb_ = glb1->ptError();
				pt2Err_glb_ = glb2->ptError();
				dpt_glb_ = glb1->pt() - glb2->pt();
				eta1_glb_ = glb1->eta();
				eta2_glb_ = glb2->eta();
				deta_glb_ = eta1_glb_ - eta2_glb_;
				norchi1_glb_ = glb1->normalizedChi2();
				norchi2_glb_ = glb2->normalizedChi2();
				
			}
			
			
			splitterTree_->Fill();
		}
	}
	
	
}


// ------------ method called once each job just before starting event loop  ------------
void CosmicSplitterValidation::beginJob()
{
	edm::LogInfo("beginJob") << "Begin Job" << std::endl;
	
	splitterTree_ = tfile->make<TTree>("splitterTree","splitterTree");
	
	// split track variables
	splitterTree_->Branch("dcaX1_spl", &dcaX1_spl_, "dcaX1_spl/D");
	splitterTree_->Branch("dcaY1_spl", &dcaY1_spl_, "dcaY1_spl/D");
	splitterTree_->Branch("dcaZ1_spl", &dcaZ1_spl_, "dcaZ1_spl/D");
	splitterTree_->Branch("dcaX2_spl", &dcaX2_spl_, "dcaX2_spl/D");
	splitterTree_->Branch("dcaY2_spl", &dcaY2_spl_, "dcaY2_spl/D");
	splitterTree_->Branch("dcaZ2_spl", &dcaZ2_spl_, "dcaZ2_spl/D");
	splitterTree_->Branch("dxy1_spl", &dxy1_spl_, "dxy1_spl/D");
	splitterTree_->Branch("dz1_spl", &dz1_spl_, "dz1_spl/D");
	splitterTree_->Branch("dxy2_spl", &dxy2_spl_, "dxy2_spl/D");
	splitterTree_->Branch("dz2_spl", &dz2_spl_, "dz2_spl/D");
	splitterTree_->Branch("theta1_spl", &theta1_spl_, "theta1_spl/D");
	splitterTree_->Branch("theta2_spl", &theta2_spl_, "theta2_spl/D");
	splitterTree_->Branch("phi1_spl", &phi1_spl_, "phi1_spl/D");
	splitterTree_->Branch("phi2_spl", &phi2_spl_, "phi2_spl/D");
	splitterTree_->Branch("ddxy_spl", &ddxy_spl_, "ddxy_spl/D");
	splitterTree_->Branch("ddz_spl", &ddz_spl_, "ddz_spl/D");
	splitterTree_->Branch("dphi_spl", &dphi_spl_, "dphi_spl/D");
	splitterTree_->Branch("dtheta_spl", &dtheta_spl_, "dtheta_spl/D");
	splitterTree_->Branch("pt1_spl", &pt1_spl_, "pt1_spl/D");
	splitterTree_->Branch("pt2_spl", &pt2_spl_, "pt2_spl/D");
	splitterTree_->Branch("dpt_spl", &dpt_spl_, "dpt_spl/D");
	splitterTree_->Branch("p1_spl", &p1_spl_, "p1_spl/D");
	splitterTree_->Branch("p2_spl", &p2_spl_, "p2_spl/D");
	splitterTree_->Branch("eta1_spl", &eta1_spl_, "eta1_spl/D");
	splitterTree_->Branch("eta2_spl", &eta2_spl_, "eta2_spl/D");
	splitterTree_->Branch("deta_spl", &deta_spl_, "deta_spl/D");
	splitterTree_->Branch("nHits1_spl", &nHits1_spl_, "nHits1_spl/D");
	splitterTree_->Branch("nHitsPXB1_spl", &nHitsPXB1_spl_, "nHitsPXB1_spl/D");
	splitterTree_->Branch("nHitsPXF1_spl", &nHitsPXF1_spl_, "nHitsPXF1_spl/D");
	splitterTree_->Branch("nHitsTIB1_spl", &nHitsTIB1_spl_, "nHitsTIB1_spl/D");
	splitterTree_->Branch("nHitsTOB1_spl", &nHitsTOB1_spl_, "nHitsTOB1_spl/D");
	splitterTree_->Branch("nHitsTID1_spl", &nHitsTID1_spl_, "nHitsTID1_spl/D");
	splitterTree_->Branch("nHitsTEC1_spl", &nHitsTEC1_spl_, "nHitsTEC1_spl/D");
	splitterTree_->Branch("nHits2_spl", &nHits2_spl_, "nHits2_spl/D");
	splitterTree_->Branch("nHitsPXB2_spl", &nHitsPXB2_spl_, "nHitsPXB2_spl/D");
	splitterTree_->Branch("nHitsPXF2_spl", &nHitsPXF2_spl_, "nHitsPXF2_spl/D");
	splitterTree_->Branch("nHitsTIB2_spl", &nHitsTIB2_spl_, "nHitsTIB2_spl/D");
	splitterTree_->Branch("nHitsTOB2_spl", &nHitsTOB2_spl_, "nHitsTOB2_spl/D");
	splitterTree_->Branch("nHitsTID2_spl", &nHitsTID2_spl_, "nHitsTID2_spl/D");
	splitterTree_->Branch("nHitsTEC2_spl", &nHitsTEC2_spl_, "nHitsTEC2_spl/D");
	
	
	splitterTree_->Branch("d01Err_spl", &d01Err_spl_, "d01Err_spl/D");
	splitterTree_->Branch("d02Err_spl", &d02Err_spl_, "d02Err_spl/D");
	splitterTree_->Branch("dz1Err_spl", &dz1Err_spl_, "dz1Err_spl/D");
	splitterTree_->Branch("dz2Err_spl", &dz2Err_spl_, "dz2Err_spl/D");
	splitterTree_->Branch("phi1Err_spl", &phi1Err_spl_, "phi1Err_spl/D");
	splitterTree_->Branch("phi2Err_spl", &phi2Err_spl_, "phi2Err_spl/D");
	splitterTree_->Branch("theta1Err_spl", &theta1Err_spl_, "theta1Err_spl/D");
	splitterTree_->Branch("theta2Err_spl", &theta2Err_spl_, "theta2Err_spl/D");
	splitterTree_->Branch("pt1Err_spl", &pt1Err_spl_, "pt1Err_spl/D");
	splitterTree_->Branch("pt2Err_spl", &pt2Err_spl_, "pt2Err_spl/D");
	
	splitterTree_->Branch("dcaX_org", &dcaX_org_, "dcaX_org/D");
	splitterTree_->Branch("dcaY_org", &dcaY_org_, "dcaY_org/D");
	splitterTree_->Branch("dcaZ_org", &dcaZ_org_, "dcaZ_org/D");
	splitterTree_->Branch("dxy_org", &dxy_org_, "dxy_org/D");
	splitterTree_->Branch("dz_org", &dz_org_, "dz_org/D");
	splitterTree_->Branch("theta_org", &theta_org_, "theta_org/D");
	splitterTree_->Branch("phi_org", &phi_org_, "phi_org/D");
	splitterTree_->Branch("eta_org", &eta_org_, "eta_org/D");
	splitterTree_->Branch("pt_org", &pt_org_, "pt_org/D");
	splitterTree_->Branch("p_org", &p_org_, "p_org/D");
	splitterTree_->Branch("norchi2_org", &norchi2_org_, "norchi2_org/D");
	
	if (splitMuons_){
		
		// standalone split 
		splitterTree_->Branch("dcaX1_sta", &dcaX1_sta_, "dcaX1_sta/D");
		splitterTree_->Branch("dcaY1_sta", &dcaY1_sta_, "dcaY1_sta/D");
		splitterTree_->Branch("dcaZ1_sta", &dcaZ1_sta_, "dcaZ1_sta/D");
		splitterTree_->Branch("dcaX2_sta", &dcaX2_sta_, "dcaX2_sta/D");
		splitterTree_->Branch("dcaY2_sta", &dcaY2_sta_, "dcaY2_sta/D");
		splitterTree_->Branch("dcaZ2_sta", &dcaZ2_sta_, "dcaZ2_sta/D");
		splitterTree_->Branch("dxy1_sta", &dxy1_sta_, "dxy1_sta/D");
		splitterTree_->Branch("dz1_sta", &dz1_sta_, "dz1_sta/D");
		splitterTree_->Branch("dxy2_sta", &dxy2_sta_, "dxy2_sta/D");
		splitterTree_->Branch("dz2_sta", &dz2_sta_, "dz2_sta/D");		
		splitterTree_->Branch("theta1_sta", &theta1_sta_, "theta1_sta/D");
		splitterTree_->Branch("theta2_sta", &theta2_sta_, "theta2_sta/D");
		splitterTree_->Branch("phi1_sta", &phi1_sta_, "phi1_sta/D");
		splitterTree_->Branch("phi2_sta", &phi2_sta_, "phi2_sta/D");
		splitterTree_->Branch("ddxy_sta", &ddxy_sta_, "ddxy_sta/D");
		splitterTree_->Branch("ddz_sta", &ddz_sta_, "ddz_sta/D");
		splitterTree_->Branch("dphi_sta", &dphi_sta_, "dphi_sta/D");
		splitterTree_->Branch("dtheta_sta", &dtheta_sta_, "dtheta_sta/D");
		splitterTree_->Branch("pt1_sta", &pt1_sta_, "pt1_sta/D");
		splitterTree_->Branch("pt2_sta", &pt2_sta_, "pt2_sta/D");
		splitterTree_->Branch("dpt_sta", &dpt_sta_, "dpt_sta/D");
		splitterTree_->Branch("p1_sta", &p1_sta_, "p1_sta/D");
		splitterTree_->Branch("p2_sta", &p2_sta_, "p2_sta/D");
		splitterTree_->Branch("eta1_sta", &eta1_sta_, "eta1_sta/D");
		splitterTree_->Branch("eta2_sta", &eta2_sta_, "eta2_sta/D");
		splitterTree_->Branch("deta_sta", &deta_sta_, "deta_sta/D");
		
		splitterTree_->Branch("d01Err_sta", &d01Err_sta_, "d01Err_sta/D");
		splitterTree_->Branch("d02Err_sta", &d02Err_sta_, "d02Err_sta/D");
		splitterTree_->Branch("dz1Err_sta", &dz1Err_sta_, "dz1Err_sta/D");
		splitterTree_->Branch("dz2Err_sta", &dz2Err_sta_, "dz2Err_sta/D");
		splitterTree_->Branch("phi1Err_sta", &phi1Err_sta_, "phi1Err_sta/D");
		splitterTree_->Branch("phi2Err_sta", &phi2Err_sta_, "phi2Err_sta/D");
		splitterTree_->Branch("theta1Err_sta", &theta1Err_sta_, "theta1Err_sta/D");
		splitterTree_->Branch("theta2Err_sta", &theta2Err_sta_, "theta2Err_sta/D");
		splitterTree_->Branch("pt1Err_sta", &pt1Err_sta_, "pt1Err_sta/D");
		splitterTree_->Branch("pt2Err_sta", &pt2Err_sta_, "pt2Err_sta/D");
		
		// global split 
		splitterTree_->Branch("dcaX1_glb", &dcaX1_glb_, "dcaX1_glb/D");
		splitterTree_->Branch("dcaY1_glb", &dcaY1_glb_, "dcaY1_glb/D");
		splitterTree_->Branch("dcaZ1_glb", &dcaZ1_glb_, "dcaZ1_glb/D");
		splitterTree_->Branch("dcaX2_glb", &dcaX2_glb_, "dcaX2_glb/D");
		splitterTree_->Branch("dcaY2_glb", &dcaY2_glb_, "dcaY2_glb/D");
		splitterTree_->Branch("dcaZ2_glb", &dcaZ2_glb_, "dcaZ2_glb/D");
		splitterTree_->Branch("dxy1_glb", &dxy1_glb_, "dxy1_glb/D");
		splitterTree_->Branch("dz1_glb", &dz1_glb_, "dz1_glb/D");
		splitterTree_->Branch("dxy2_glb", &dxy2_glb_, "dxy2_glb/D");
		splitterTree_->Branch("dz2_glb", &dz2_glb_, "dz2_glb/D");		
		splitterTree_->Branch("theta1_glb", &theta1_glb_, "theta1_glb/D");
		splitterTree_->Branch("theta2_glb", &theta2_glb_, "theta2_glb/D");
		splitterTree_->Branch("phi1_glb", &phi1_glb_, "phi1_glb/D");
		splitterTree_->Branch("phi2_glb", &phi2_glb_, "phi2_glb/D");
		splitterTree_->Branch("ddxy_glb", &ddxy_glb_, "ddxy_glb/D");
		splitterTree_->Branch("ddz_glb", &ddz_glb_, "ddz_glb/D");
		splitterTree_->Branch("dphi_glb", &dphi_glb_, "dphi_glb/D");
		splitterTree_->Branch("dtheta_glb", &dtheta_glb_, "dtheta_glb/D");
		splitterTree_->Branch("pt1_glb", &pt1_glb_, "pt1_glb/D");
		splitterTree_->Branch("pt2_glb", &pt2_glb_, "pt2_glb/D");
		splitterTree_->Branch("dpt_glb", &dpt_glb_, "dpt_glb/D");
		splitterTree_->Branch("p1_glb", &p1_glb_, "p1_glb/D");
		splitterTree_->Branch("p2_glb", &p2_glb_, "p2_glb/D");
		splitterTree_->Branch("eta1_glb", &eta1_glb_, "eta1_glb/D");
		splitterTree_->Branch("eta2_glb", &eta2_glb_, "eta2_glb/D");
		splitterTree_->Branch("deta_glb", &deta_glb_, "deta_glb/D");
		splitterTree_->Branch("norchi1_glb", &norchi1_glb_, "norchi1_glb/D");
		splitterTree_->Branch("norchi2_glb", &norchi2_glb_, "norchi2_glb/D");

		splitterTree_->Branch("d01Err_glb", &d01Err_glb_, "d01Err_glb/D");
		splitterTree_->Branch("d02Err_glb", &d02Err_glb_, "d02Err_glb/D");
		splitterTree_->Branch("dz1Err_glb", &dz1Err_glb_, "dz1Err_glb/D");
		splitterTree_->Branch("dz2Err_glb", &dz2Err_glb_, "dz2Err_glb/D");
		splitterTree_->Branch("phi1Err_glb", &phi1Err_glb_, "phi1Err_glb/D");
		splitterTree_->Branch("phi2Err_glb", &phi2Err_glb_, "phi2Err_glb/D");
		splitterTree_->Branch("theta1Err_glb", &theta1Err_glb_, "theta1Err_glb/D");
		splitterTree_->Branch("theta2Err_glb", &theta2Err_glb_, "theta2Err_glb/D");
		splitterTree_->Branch("pt1Err_glb", &pt1Err_glb_, "pt1Err_glb/D");
		splitterTree_->Branch("pt2Err_glb", &pt2Err_glb_, "pt2Err_glb/D");

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
