// -*- C++ -*-
//
// Package:    TrackerGeometryIntoNtuples
// Class:      TrackerGeometryIntoNtuples
// 
/**\class TrackerGeometryIntoNtuples TrackerGeometryIntoNtuples.cc 
 
 Description: Takes a set of alignment constants and turns them into a ROOT file
 
 Implementation:
 <Notes on implementation>
 */
//
// Original Author:  Nhan Tran
//         Created:  Mon Jul 16m 16:56:34 CDT 2007
// $Id: TrackerGeometryIntoNtuples.cc,v 1.9 2012/06/13 09:20:14 yana Exp $
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

#include "CLHEP/Matrix/SymMatrix.h"

//
// class decleration
//

class TrackerGeometryIntoNtuples : public edm::EDAnalyzer {
public:
	explicit TrackerGeometryIntoNtuples(const edm::ParameterSet&);
	~TrackerGeometryIntoNtuples();
	
	
private:
	virtual void analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup);
	
	void addBranches();
	
	// ----------member data ---------------------------
	//std::vector<AlignTransform> m_align;
	AlignableTracker* theCurrentTracker;
	
	uint32_t m_rawid;
	double m_x, m_y, m_z;
	double m_alpha, m_beta, m_gamma;
	int m_subdetid;
	double m_xx, m_xy, m_yy, m_xz, m_yz, m_zz;
	
	TTree *m_tree;
	TTree *m_treeErrors;
	std::string m_outputFile;
	std::string m_outputTreename;
	TFile *m_file;

  const edm::ParameterSet theParameterSet;
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
TrackerGeometryIntoNtuples::TrackerGeometryIntoNtuples(const edm::ParameterSet& iConfig) :
  theCurrentTracker(0),
  m_rawid(0),
  m_x(0.), m_y(0.), m_z(0.),
  m_alpha(0.), m_beta(0.), m_gamma(0.),
  m_subdetid(0),
  m_xx(0.), m_xy(0.), m_yy(0.), m_xz(0.), m_yz(0.), m_zz(0.),
  theParameterSet( iConfig )
{
	m_outputFile = iConfig.getUntrackedParameter< std::string > ("outputFile");
	m_outputTreename = iConfig.getUntrackedParameter< std::string > ("outputTreename");
	m_file = new TFile(m_outputFile.c_str(),"RECREATE");
	m_tree = new TTree(m_outputTreename.c_str(),m_outputTreename.c_str());
	//char errorTreeName[256];
	//snprintf(errorTreeName, sizeof(errorTreeName), "%sErrors", m_outputTreename);
	//m_treeErrors = new TTree(errorTreeName,errorTreeName);
	m_treeErrors = new TTree("alignTreeErrors","alignTreeErrors");
}


TrackerGeometryIntoNtuples::~TrackerGeometryIntoNtuples()
{
  delete theCurrentTracker;
}


//
// member functions
//

// ------------ method called to for each event  ------------
void TrackerGeometryIntoNtuples::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	edm::LogInfo("beginJob") << "Begin Job" << std::endl;
	
	//accessing the initial geometry
	edm::ESHandle<GeometricDet> theGeometricDet;
	iSetup.get<IdealGeometryRecord>().get(theGeometricDet);
	TrackerGeomBuilderFromGeometricDet trackerBuilder;
	//currernt tracker
	const edm::ParameterSet tkGeomConsts( theParameterSet.getParameter<edm::ParameterSet>( "trackerGeometryConstants" ));
	TrackerGeometry* theCurTracker = trackerBuilder.build(&*theGeometricDet,
							      tkGeomConsts.getParameter<bool>("upgradeGeometry"),
							      tkGeomConsts.getParameter<int>( "ROWS_PER_ROC" ),
							      tkGeomConsts.getParameter<int>( "COLS_PER_ROC" ),
							      tkGeomConsts.getParameter<int>( "BIG_PIX_PER_ROC_X" ),
							      tkGeomConsts.getParameter<int>( "BIG_PIX_PER_ROC_Y" ),
							      tkGeomConsts.getParameter<int>( "ROCS_X" ),
							      tkGeomConsts.getParameter<int>( "ROCS_Y" ));
	
	//build the tracker
	edm::ESHandle<Alignments> alignments;
	edm::ESHandle<AlignmentErrors> alignmentErrors;
	
	iSetup.get<TrackerAlignmentRcd>().get(alignments);
	iSetup.get<TrackerAlignmentErrorRcd>().get(alignmentErrors);
	
	//apply the latest alignments
	edm::ESHandle<Alignments> globalPositionRcd;
	iSetup.get<TrackerDigiGeometryRecord>().getRecord<GlobalPositionRcd>().get(globalPositionRcd);
	GeometryAligner aligner;
	aligner.applyAlignments<TrackerGeometry>( &(*theCurTracker), &(*alignments), &(*alignmentErrors),
											 align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Tracker)));
	
	
	theCurrentTracker = new AlignableTracker(&(*theCurTracker));	
	
	Alignments* theAlignments = theCurrentTracker->alignments();
	//AlignmentErrors* theAlignmentErrors = theCurrentTracker->alignmentErrors();	
	
	//alignments
	addBranches();
	for (std::vector<AlignTransform>::const_iterator i = theAlignments->m_align.begin(); i != theAlignments->m_align.end(); ++i){
		
		m_rawid = i->rawId();
		CLHEP::Hep3Vector translation = i->translation();
		m_x = translation.x();
		m_y = translation.y();
		m_z = translation.z();
		
	
		CLHEP::HepRotation rotation = i->rotation();
		m_alpha = rotation.getPhi();
		m_beta = rotation.getTheta();
		m_gamma = rotation.getPsi();
		m_tree->Fill();
		
		//DetId detid(m_rawid);
		//if (detid.subdetId() > 2){
		//PXFDetId pxfid( m_rawid );
		//std::cout << " panel: " << pxfid.panel() << ", module: " << pxfid.module() << std::endl;
		//if ((pxfid.panel() == 1) && (pxfid.module() == 4)) std::cout << m_rawid << ", ";
		//std::cout << m_rawid << std::setprecision(9) <<  " " << m_x << " " << m_y << " " << m_z;
		//std::cout << std::setprecision(9) << " " << m_alpha << " " << m_beta << " " << m_gamma << std::endl;  
		//}
		
	}
	
	delete theAlignments;

	std::vector<AlignTransformError> alignErrors = alignmentErrors->m_alignError;
	for (std::vector<AlignTransformError>::const_iterator i = alignErrors.begin(); i != alignErrors.end(); ++i){

		m_rawid = i->rawId();
		CLHEP::HepSymMatrix errMatrix = i->matrix();
		DetId detid(m_rawid);
		m_subdetid = detid.subdetId();
		m_xx = errMatrix[0][0];
		m_xy = errMatrix[0][1];
		m_xz = errMatrix[0][2];
		m_yy = errMatrix[1][1];
		m_yz = errMatrix[1][2];
		m_zz = errMatrix[2][2];
		m_treeErrors->Fill();
	}
	
	//write out 
	m_file->cd();
	m_tree->Write();
	m_treeErrors->Write();
	m_file->Close();
}


void TrackerGeometryIntoNtuples::addBranches() {
	
	m_tree->Branch("rawid", &m_rawid, "rawid/I");
	m_tree->Branch("x", &m_x, "x/D");
	m_tree->Branch("y", &m_y, "y/D");
	m_tree->Branch("z", &m_z, "z/D");
	m_tree->Branch("alpha", &m_alpha, "alpha/D");
	m_tree->Branch("beta", &m_beta, "beta/D");
	m_tree->Branch("gamma", &m_gamma, "gamma/D");
	
	
	m_treeErrors->Branch("rawid", &m_rawid, "rawid/I");
	m_treeErrors->Branch("subdetid", &m_subdetid, "subdetid/I");
	m_treeErrors->Branch("xx", &m_xx, "xx/D");
	m_treeErrors->Branch("yy", &m_yy, "yy/D");
	m_treeErrors->Branch("zz", &m_zz, "zz/D");
	m_treeErrors->Branch("xy", &m_xy, "xy/D");
	m_treeErrors->Branch("xz", &m_xz, "xz/D");
	m_treeErrors->Branch("yz", &m_yz, "yz/D");
		
}


//define this as a plug-in
DEFINE_FWK_MODULE(TrackerGeometryIntoNtuples);
