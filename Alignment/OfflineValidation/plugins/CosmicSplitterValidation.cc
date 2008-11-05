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
// $Id: CosmicSplitterValidation.cc,v 1.1 2008/09/23 14:02:05 ntran Exp $
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


#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TrajectoryState/interface/CopyUsingClone.h" 
#include "RecoVertex/VertexTools/interface/PerigeeLinearizedTrackState.h" 
#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryPlugin.h"



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
	
	edm::InputTag tracks_;
	bool checkIfGolden_;
	int totalTracksToAnalyzer_;
	int goldenCtr;
	int twoTracksCtr;
	int goldenPlusTwoTracksCtr;
	edm::ESHandle<MagneticField>   theMagField;
	
	
	edm::Service<TFileService> tfile;
	// ----------member data ---------------------------
	//std::vector<AlignTransform> m_align;
	//HISTOGRAMS
	//multiplicity
	TH1F* h_theta;
	TH1F* h_phi;
	TH1F* h_dxy;
	TH1F* h_dz;
	TH1F* h_dpt;
	TTree* _splitterTree;
	// tree vars
	//int _hits1, _hits2;
	double _dcaX1, _dcaY1, _dcaZ1;
	double _dcaX2, _dcaY2, _dcaZ2;
	double _theta1, _theta2, _phi1, _phi2;
	double _dxy, _dz, _dtheta, _dphi;
	double _pt1, _pt2, _dpt;
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
tracks_(iConfig.getParameter<edm::InputTag>("tracks")),
checkIfGolden_(iConfig.getParameter<bool>("checkIfGolden"))
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
	//if (isGolden) std::cout << "FOUND A GOLDEN MUON EVENT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
	
	iSetup.get<IdealMagneticFieldRecord>().get(theMagField);
	
	// read with View, so we can read also a TrackRefVector
    edm::Handle<std::vector<reco::Track> > tracks;
    iEvent.getByLabel(tracks_, tracks);
	
	std::cout << "Number of tracks: " << tracks->size() << std::endl;
	totalTracksToAnalyzer_ = totalTracksToAnalyzer_ + tracks->size();
	if (isGolden) goldenCtr++;
	if (tracks->size() == 2) twoTracksCtr++;
	if (tracks->size() == 2 && isGolden){
		goldenPlusTwoTracksCtr++;
		
		reco::Track track1 = tracks->at(0);
		reco::Track track2 = tracks->at(1);
		
		// find dca
		// track1
		//reco::TransientTrack tt1( track1, theMagField.product() );//, theGeometry);
		//FreeTrajectoryState fts1 = tt1.initialFreeState();
		//TSCPBuilderNoMaterial tscpBuilder1;
		//TrajectoryStateClosestToPoint tsAtClosestApproach1 = tscpBuilder1(fts1,GlobalPoint(0,0,0));//as in TrackProducerAlgorithm
		//GlobalPoint dca1 = tsAtClosestApproach1.theState().position();
		math::XYZPoint dca1 = track1.referencePoint();
		//std::cout << "analyzer: reference point1: " << refPoint1 << ", ref point the long way: " << dca1 << std::endl;
		// track2
		//reco::TransientTrack tt2( track2, theMagField.product() );//, theGeometry);
		//FreeTrajectoryState fts2 = tt2.initialFreeState();
		//TSCPBuilderNoMaterial tscpBuilder2;
		//TrajectoryStateClosestToPoint tsAtClosestApproach2 = tscpBuilder2(fts2,GlobalPoint(0,0,0));//as in TrackProducerAlgorithm
		//GlobalPoint dca2 = tsAtClosestApproach2.theState().position();
		math::XYZPoint dca2 = track2.referencePoint();
		//std::cout << "analyzer: reference point2: " << refPoint2 << ", ref point the long way: " << dca2 << std::endl;

		
		double dtheta_Val = track1.theta() - track2.theta();
		double dphi_Val = track1.phi() - track2.phi();
		//double dxy1 = sqrt( dca1.x()*dca1.x() + dca1.y()*dca1.y() );
		//double dxy2 = sqrt( dca1.x()*dca1.x() + dca2.y()*dca2.y() );
		//double dxy_Val = dxy1 - dxy2;
		double dxy_Val = track1.d0() - track2.d0();
		//double dz_Val = dca1.z() - dca2.z();
		double dz_Val = track1.dz() - track2.dz();
		double dpt_Val = track1.pt() - track2.pt();
		
		// fill histos
		h_theta->Fill( dtheta_Val );
		h_phi->Fill( dphi_Val );
		h_dxy->Fill( dxy_Val );
		h_dz->Fill( dz_Val );
		h_dpt->Fill( dpt_Val );
		
		// fill tree
		//int _hits1, _hits2;
		_dcaX1 = dca1.x(); 
		_dcaY1 = dca1.y();
		_dcaZ1 = dca1.z();
		_dcaX2 = dca2.x(); 
		_dcaY2 = dca2.y();
		_dcaZ2 = dca2.z();		
		_theta1 = track1.theta();
		_theta2 = track2.theta();
		_phi1 = track1.phi();
		_phi2 = track2.phi();
		_dxy = dxy_Val;
		_dz = dz_Val;
		_dtheta = dtheta_Val;
		_dphi = dphi_Val;
		_pt1 = track1.pt();
		_pt2 = track2.pt();
		_dpt = dpt_Val;
		_splitterTree->Fill();
	}
	
	
}


// ------------ method called once each job just before starting event loop  ------------
void CosmicSplitterValidation::beginJob(const edm::EventSetup& iSetup)
{
	edm::LogInfo("beginJob") << "Begin Job" << std::endl;
	h_theta = tfile->make<TH1F>("h_theta","#Delta #Theta",400,-1,1);
	h_phi = tfile->make<TH1F>("h_phi","#Delta #phi",400,-0.2,0.2);
	h_dxy = tfile->make<TH1F>("h_dxy", "#Delta dxy", 400, -2, 2);
	h_dz = tfile->make<TH1F>("h_dz", "#Delta dz", 400, -50, 50);
	h_dpt = tfile->make<TH1F>("h_dpt", "#Delta pt", 400, -20, 20);
	
	_splitterTree = tfile->make<TTree>("splitterTree","splitterTree");
	//_splitterTree->Branch("hits1", &_hits1, "hits1/I");
	//_splitterTree->Branch("hits2", &_hits2, "hits2/I");
	_splitterTree->Branch("dcaX1", &_dcaX1, "dcaX1/D");
	_splitterTree->Branch("dcaY1", &_dcaY1, "dcaY1/D");
	_splitterTree->Branch("dcaZ1", &_dcaZ1, "dcaZ1/D");
	_splitterTree->Branch("dcaX2", &_dcaX2, "dcaX2/D");
	_splitterTree->Branch("dcaY2", &_dcaY2, "dcaY2/D");
	_splitterTree->Branch("dcaZ2", &_dcaZ2, "dcaZ2/D");
	_splitterTree->Branch("theta1", &_theta1, "theta1/D");
	_splitterTree->Branch("theta2", &_theta2, "theta2/D");
	_splitterTree->Branch("phi1", &_phi1, "phi1/D");
	_splitterTree->Branch("phi2", &_phi2, "phi2/D");
	_splitterTree->Branch("dxy", &_dxy, "dxy/D");
	_splitterTree->Branch("dz", &_dz, "dz/D");
	_splitterTree->Branch("dphi", &_dphi, "dphi/D");
	_splitterTree->Branch("dtheta", &_dtheta, "dtheta/D");
	_splitterTree->Branch("pt1", &_pt1, "pt1/D");
	_splitterTree->Branch("pt2", &_pt2, "pt2/D");
	_splitterTree->Branch("dpt", &_dpt, "dpt/D");
	
	
	totalTracksToAnalyzer_ = 0;
	goldenCtr = 0;
	twoTracksCtr = 0;
	goldenPlusTwoTracksCtr = 0;
}


// ------------ method called once each job just after ending the event loop  ------------
void CosmicSplitterValidation::endJob() {
	
	//std::cout << "totalTracksToAnalyzer: " << totalTracksToAnalyzer_ << std::endl;
	std::cout << "golden: " << goldenCtr << ", two tracks: " << twoTracksCtr << ", golden+twotracks: " << goldenPlusTwoTracksCtr << std::endl;
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
