// -*- C++ -*-
//
// Package:     CommonAlignmentProducer
// Class  :     AlignmentMonitorSurvey
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Nhan Tran
//         Created:  10/8/07
// $Id: SWGuideAlignmentMonitors.txt,v 1.8 2007/07/19 20:22:29 pivarski Exp $
//

#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/SurveyResidual.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "CondFormats/Alignment/interface/SurveyErrors.h"
#include "CondFormats/DataRecord/interface/TrackerSurveyRcd.h"
#include "CondFormats/DataRecord/interface/TrackerSurveyErrorRcd.h"

#include "TTree.h"

class AlignmentMonitorSurvey: public AlignmentMonitorBase {
public:
	AlignmentMonitorSurvey(const edm::ParameterSet& cfg);
	~AlignmentMonitorSurvey() {};
	
	void book();
	void event(const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& iTrajTracks);
	void afterAlignment(const edm::EventSetup &iSetup);
	
private:

	//Histogram parameters
	edm::ParameterSet m_params;
	bool m_params_useResid;
	
	//Trees
	TTree *m_before;
	TTree *m_after;
	Int_t m_before_rawid, m_before_level;
	Float_t m_before_dx, m_before_dy, m_before_dz;//, m_before_phix, m_before_phiy, m_before_phiz;
	Float_t m_before_resx, m_before_resy, m_before_resz, m_before_resphix, m_before_resphiy, m_before_resphiz;
	Int_t m_after_rawid, m_after_level;
	Float_t m_after_dx, m_after_dy, m_after_dz;// m_after_phix, m_after_phiy, m_after_phiz;
	Float_t m_after_resx, m_after_resy, m_after_resz, m_after_resphix, m_after_resphiy, m_after_resphiz;


	//Alignment Stuff
	
	std::vector<Alignable*> alignables_before;
	std::vector<Alignable*> alignables_after;


	const Alignments* theSurveyValues;
	const SurveyErrors* theSurveyErrors;

};



AlignmentMonitorSurvey::AlignmentMonitorSurvey(const edm::ParameterSet& cfg)
	:AlignmentMonitorBase(cfg)
	 , m_params(cfg.getParameter<edm::ParameterSet>("params"))
{
	m_params_useResid = m_params.getParameter<bool>("useResiduals");
}


void AlignmentMonitorSurvey::book() {

	
	
	edm::LogInfo("AlignmentMonitorSurvey") << "[AlignmentMonitorSurvey] Monitor beforeAlignment";


	m_before = (TTree*)(add("/iterN/", new TTree("before", "positions before iteration")));
	m_before->Branch("rawid", &m_before_rawid, "rawid/I");
	m_before->Branch("level", &m_before_level, "level/I");
	m_before->Branch("dx", &m_before_dx, "dx/F");
	m_before->Branch("dy", &m_before_dy, "dy/F");
	m_before->Branch("dz", &m_before_dz, "dz/F");
	m_before->Branch("resx", &m_before_resx, "resx/F");
	m_before->Branch("resy", &m_before_resy, "resy/F");
	m_before->Branch("resz", &m_before_resz, "resz/F");
	m_before->Branch("resphix", &m_before_resphix, "resphix/F");
	m_before->Branch("resphiy", &m_before_resphiy, "resphiy/F");
	m_before->Branch("resphiz", &m_before_resphiz, "resphiz/F");
	m_after = (TTree*)(add("/iterN/", new TTree("after", "positions after iteration")));
	m_after->Branch("rawid", &m_after_rawid, "rawid/I");
	m_after->Branch("level", &m_after_level, "level/I");
	m_after->Branch("dx", &m_after_dx, "dx/F");
	m_after->Branch("dy", &m_after_dy, "dy/F");
	m_after->Branch("dz", &m_after_dz, "dz/F");
	m_after->Branch("resx", &m_after_resx, "resx/F");
	m_after->Branch("resy", &m_after_resy, "resy/F");
	m_after->Branch("resz", &m_after_resz, "resz/F");
	m_after->Branch("resphix", &m_after_resphix, "resphix/F");
	m_after->Branch("resphiy", &m_after_resphiy, "resphiy/F");
	m_after->Branch("resphiz", &m_after_resphiz, "resphiz/F");
	
	//using information in "before" Parameter Store
	alignables_before = pStore()->alignables();
	
	int it = 0;
	for (std::vector<Alignable*>::const_iterator aliiter = alignables_before.begin();  aliiter != alignables_before.end();  ++aliiter) {

		Alignable *ali = *aliiter;
		m_before_rawid = ali->id();
		m_before_level = ali->alignableObjectId();
		
		//displacements
		m_before_dx = ali->displacement().x();
		m_before_dy = ali->displacement().y();
		m_before_dz = ali->displacement().z();
		
		if (m_params_useResid){
			AlignableObjectId::AlignableObjectIdType objId = (AlignableObjectId::AlignableObjectIdType) ali->alignableObjectId();
			SurveyResidual resid(*ali, objId);
			AlgebraicVector resParams = resid.sensorResidual();
			
			m_before_resx = resParams[0];
			m_before_resy = resParams[1];
			m_before_resz = resParams[2];
			m_before_resphix = resParams[3];
			m_before_resphiy = resParams[4];
			m_before_resphiz = resParams[5];
		}	
		m_before->Fill();
		it++;
	}


	
}

void AlignmentMonitorSurvey::event(const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& tracks) {
}

void AlignmentMonitorSurvey::afterAlignment(const edm::EventSetup &iSetup) {

	edm::LogInfo("AlignmentMonitorSurvey") << "[AlignmentMonitorSurvey] Monitor afterAlignment";
		
	alignables_after = pStore()->alignables();
	for (std::vector<Alignable*>::const_iterator aliiter = alignables_after.begin();  aliiter != alignables_after.end();  ++aliiter) {

		Alignable *ali = *aliiter;
		m_after_rawid = ali->id();
		m_after_level = ali->alignableObjectId();

		//displacements
		m_after_dx = ali->displacement().x();
		m_after_dy = ali->displacement().y();
		m_after_dz = ali->displacement().z();
		
		if(m_params_useResid){
			AlignableObjectId::AlignableObjectIdType objId = (AlignableObjectId::AlignableObjectIdType) ali->alignableObjectId();
			SurveyResidual resid(*ali, objId);
			AlgebraicVector resParams = resid.sensorResidual();
			
			m_after_resx = resParams[0];
			m_after_resy = resParams[1];
			m_after_resz = resParams[2];
			m_after_resphix = resParams[3];
			m_after_resphiy = resParams[4];
			m_after_resphiz = resParams[5];
		}
		
		m_after->Fill();
	}
}

DEFINE_EDM_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorSurvey, "AlignmentMonitorSurvey");
