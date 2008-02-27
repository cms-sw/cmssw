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
// $Id: TrackerGeometryIntoNtuples.cc,v 1.5 2008/02/21 12:03:16 flucke Exp $
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

//
// class decleration
//

class TrackerGeometryIntoNtuples : public edm::EDAnalyzer {
public:
	explicit TrackerGeometryIntoNtuples(const edm::ParameterSet&);
	~TrackerGeometryIntoNtuples();
	
	
private:
	virtual void beginJob(const edm::EventSetup &iSetup);
	virtual void analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup);
	virtual void endJob() ;
	
	void addBranches();
	
	// ----------member data ---------------------------
	//std::vector<AlignTransform> m_align;
	Alignments* theAlignments;
	AlignableTracker* theCurrentTracker;

	uint32_t m_rawid;
	double m_x, m_y, m_z;
	double m_alpha, m_beta, m_gamma;
	double m_xx, m_yx, m_yy, m_zx, m_zy, m_zz;
	
	TTree *m_tree;
	TTree *m_treeErrors;
	std::string m_outputFile;
	std::string m_outputTreename;
	TFile *m_file;
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
TrackerGeometryIntoNtuples::TrackerGeometryIntoNtuples(const edm::ParameterSet& iConfig)
{
	m_outputFile = iConfig.getUntrackedParameter< std::string > ("outputFile");
	m_outputTreename = iConfig.getUntrackedParameter< std::string > ("outputTreename");
	m_file = new TFile(m_outputFile.c_str(),"RECREATE");
	m_tree = new TTree(m_outputTreename.c_str(),m_outputTreename.c_str());

	
}


TrackerGeometryIntoNtuples::~TrackerGeometryIntoNtuples()
{}


//
// member functions
//

// ------------ method called to for each event  ------------
void TrackerGeometryIntoNtuples::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{}


// ------------ method called once each job just before starting event loop  ------------
void TrackerGeometryIntoNtuples::beginJob(const edm::EventSetup& iSetup)
{
	edm::LogInfo("beginJob") << "Begin Job" << std::endl;
	
	//accessing the initial geometry
	edm::ESHandle<GeometricDet> theGeometricDet;
	iSetup.get<IdealGeometryRecord>().get(theGeometricDet);
	TrackerGeomBuilderFromGeometricDet trackerBuilder;
	//currernt tracker
	TrackerGeometry* theCurTracker = trackerBuilder.build(&*theGeometricDet); 

	
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
	//not used AlignmentErrors* theAlignmentErrors = theCurrentTracker->alignmentErrors();	
	//alignments
	addBranches();
	for (std::vector<AlignTransform>::const_iterator i = theAlignments->m_align.begin(); i != theAlignments->m_align.end(); ++i){

		m_rawid = i->rawId();
		Hep3Vector translation = i->translation();
		m_x = translation.x();
		m_y = translation.y();
		m_z = translation.z();

		
		HepRotation rotation = i->rotation();
		m_alpha = rotation.getPhi();
		m_beta = rotation.getTheta();
		m_gamma = rotation.getPsi();
		m_tree->Fill();
		
	}

	//write out 
	m_file->cd();
	m_tree->Write();
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

}


// ------------ method called once each job just after ending the event loop  ------------
void TrackerGeometryIntoNtuples::endJob() {
}



//define this as a plug-in
DEFINE_FWK_MODULE(TrackerGeometryIntoNtuples);
