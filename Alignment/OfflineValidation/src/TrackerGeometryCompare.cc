#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/SurveyErrors.h"
#include "CondFormats/DataRecord/interface/TrackerSurveyRcd.h"
#include "CondFormats/DataRecord/interface/TrackerSurveyErrorRcd.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentErrorRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/SurveyResidual.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignTools.h"
#include "CondFormats/Alignment/interface/SurveyErrors.h"
#include "CondFormats/DataRecord/interface/TrackerSurveyRcd.h"
#include "CondFormats/DataRecord/interface/TrackerSurveyErrorRcd.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Alignment/OfflineValidation/interface/TrackerGeometryCompare.h"


TrackerGeometryCompare::TrackerGeometryCompare(const edm::ParameterSet& cfg)
{
	_filename = cfg.getUntrackedParameter< std::string > ("outputFile");
	_inputType = cfg.getUntrackedParameter< std::string > ("inputType");
	AlignableObjectId dummy;
	const std::vector<std::string>& levels = cfg.getUntrackedParameter< std::vector<std::string> > ("levels");
	const std::vector<int>& subdets = cfg.getUntrackedParameter< std::vector<int> > ("subdets");
	edm::LogInfo("TrakcerGeomertyCompare") << "levels: " << levels.size();
	for (unsigned int l = 0; l < levels.size(); ++l){
		theLevels.push_back( dummy.nameToType(levels[l]));
		edm::LogInfo("TrakcerGeomertyCompare") << "level: " << levels[l];
	}
	for (unsigned int l = 0; l < levels.size(); ++l){
		theSubDets.push_back( subdets[l] );
	}
	

	
	//root configuration
	_theFile = new TFile(_filename.c_str(),"RECREATE");
	_alignTree = new TTree("alignTree","alignTree");//,"id:level:mid:mlevel:sublevel:x:y:z:r:phi:a:b:c:dx:dy:dz:dr:dphi:da:db:dc");
	_alignTree->Branch("id", &_id, "id/I");
	_alignTree->Branch("level", &_level, "level/I");
	_alignTree->Branch("mid", &_mid, "mid/I");
	_alignTree->Branch("mlevel", &_mlevel, "mlevel/I");
	_alignTree->Branch("sublevel", &_sublevel, "sublevel/I");
	_alignTree->Branch("x", &_xVal, "x/F");
	_alignTree->Branch("y", &_yVal, "y/F");
	_alignTree->Branch("z", &_zVal, "z/F");
	_alignTree->Branch("r", &_rVal, "r/F");
	_alignTree->Branch("phi", &_phiVal, "phi/F");
	_alignTree->Branch("alpha", &_alphaVal, "alpha/F");
	_alignTree->Branch("beta", &_betaVal, "beta/F");
	_alignTree->Branch("gamma", &_gammaVal, "gamma/F");
	_alignTree->Branch("dx", &_dxVal, "dx/F");
	_alignTree->Branch("dy", &_dyVal, "dy/F");
	_alignTree->Branch("dz", &_dzVal, "dz/F");
	_alignTree->Branch("dr", &_drVal, "dr/F");
	_alignTree->Branch("dphi", &_dphiVal, "dphi/F");
	_alignTree->Branch("dalpha", &_dalphaVal, "dalpha/F");
	_alignTree->Branch("dbeta", &_dbetaVal, "dbeta/F");
	_alignTree->Branch("dgamma", &_dgammaVal, "dgamma/F");
}

void TrackerGeometryCompare::beginJob(const edm::EventSetup& iSetup){

	typedef AlignTransform SurveyValue;
	typedef Alignments SurveyValues;
	
	//accessing the initial geometry
	edm::ESHandle<DDCompactView> cpv;
	iSetup.get<IdealGeometryRecord>().get(cpv);
	edm::ESHandle<GeometricDet> theGeometricDet;
	iSetup.get<IdealGeometryRecord>().get(theGeometricDet);
	TrackerGeomBuilderFromGeometricDet trackerBuilder;
	//reference tracker
	TrackerGeometry* theRefTracker = trackerBuilder.build(&*cpv, &*theGeometricDet); 
	referenceTracker = new AlignableTracker(&(*theGeometricDet),&(*theRefTracker));
	//currernt tracker
	TrackerGeometry* theCurTracker = trackerBuilder.build(&*cpv, &*theGeometricDet); 


	if (_inputType == "tracker"){
		edm::ESHandle<Alignments> alignments;
		edm::ESHandle<AlignmentErrors> alignmentErrors;
		
		iSetup.get<TrackerAlignmentRcd>().get(alignments);
		iSetup.get<TrackerAlignmentErrorRcd>().get(alignmentErrors);
		
		//apply the latest alignments
		GeometryAligner aligner;
		aligner.applyAlignments<TrackerGeometry>( &(*theCurTracker), &(*alignments), &(*alignmentErrors));
		currentTracker = new AlignableTracker(&(*theGeometricDet),&(*theCurTracker));
		
		compareGeometries(referenceTracker,currentTracker);
				
	}
	if (_inputType == "survey"){
		edm::ESHandle<SurveyValues> valuesHandle;
		edm::ESHandle<SurveyErrors> errorsHandle;

		iSetup.get<TrackerSurveyRcd>().get(valuesHandle);
		iSetup.get<TrackerSurveyErrorRcd>().get(errorsHandle);

		const std::vector<SurveyValue>& values = valuesHandle->m_align;
		const std::vector<SurveyError>& errors = errorsHandle->m_surveyErrors;
		
		unsigned int size = values.size();
		
		for (unsigned int i = 0; i < size; ++i)
		{
			const SurveyValue& value = values[i];
			const SurveyError& error = errors[i];
			
			edm::LogInfo("SurveyDBReader")
				<< "Type " << static_cast<unsigned int>( error.structureType() )
				<< " raw id " << error.rawId()
				<< " pos " << value.translation()
				<< " rot " << value.rotation();
		}
		
		edm::LogInfo("SurveyDBReader")
			<< "Number of alignables read " << size << std::endl;
		
	}
	
	//write out ntuple
	//might be better to do within output module
	_theFile->cd();
	_alignTree->Write();
	_theFile->Close();
	
}

void TrackerGeometryCompare::analyze(const edm::Event&, const edm::EventSetup& iSetup){

}

void TrackerGeometryCompare::compareGeometries(Alignable* refAli, Alignable* curAli){

	const std::vector<Alignable*>& refComp = refAli->components();
	const std::vector<Alignable*>& curComp = curAli->components();
	unsigned int nComp = refComp.size();

	//only perform for designate levels
	bool useLevel = false;
	for (unsigned int i = 0; i < theLevels.size(); ++i){
		if (refAli->alignableObjectId() == theLevels[i]) useLevel = true;
	}

	//set predefined sublevels
	for (unsigned int i = 0; i < theLevels.size(); ++i){
		DetId detid(refAli->id());
		if (detid.subdetId() == theSubDets[i]) useLevel = true;
	}
	//useLevel = true;
	if (useLevel){
		AlgebraicVector diff = align::diffAlignables(refAli,curAli);
		fillTree(refAli, diff);
		align::moveAlignable(curAli, diff);
		float tolerance = 1e-7;
		diff = align::diffAlignables(refAli,curAli);
		align::GlobalVector check1(diff[0],diff[1],diff[2]);
		align::GlobalVector check2(diff[3],diff[4],diff[5]);
		DetId detid(refAli->id());
		if ((check1.mag() > tolerance)||(check2.mag() > tolerance)){
			edm::LogWarning("CopareGeoms") << "Tolerance Exceeded!(alObjId - " << refAli->alignableObjectId() << ", subdetId - "<< detid.subdetId() << ")";
		}
	}
	//need a workaround for it right now, until the hierarchy is created...
	/*
	if (refAli->alignableObjectId() == 19){
	  AlgebraicVector diff;
		diff = align::diffAlignables(refComp[0],curComp[0]); align::moveAlignable(curComp[0], diff);
		edm::LogInfo("test") << "diffcheck: " << align::diffAlignables(refComp[0],curComp[0]);
		diff = align::diffAlignables(refComp[1],curComp[1]); align::moveAlignable(curComp[1], diff);
		edm::LogInfo("test") << "diffcheck: " << align::diffAlignables(refComp[1],curComp[1]);
		diff = align::diffAlignables(refComp[2],curComp[2]); align::moveAlignable(curComp[2], diff);
		edm::LogInfo("test") << "diffcheck: " << align::diffAlignables(refComp[2],curComp[2]);
		diff = align::diffAlignables(refComp[3],curComp[3]); align::moveAlignable(curComp[3], diff);
		edm::LogInfo("test") << "diffcheck: " << align::diffAlignables(refComp[3],curComp[3]);
		diff = align::diffAlignables(refComp[8],curComp[8]); align::moveAlignable(curComp[8], diff);
		edm::LogInfo("test") << "diffcheck: " << align::diffAlignables(refComp[8],curComp[8]);
		diff = align::diffAlignables(refComp[9],curComp[9]); align::moveAlignable(curComp[9], diff);
		edm::LogInfo("test") << "diffcheck: " << align::diffAlignables(refComp[9],curComp[9]);
		diff = align::diffAlignables(refComp[10],curComp[10]); align::moveAlignable(curComp[10], diff);
		edm::LogInfo("test") << "diffcheck: " << align::diffAlignables(refComp[10],curComp[10]);
		diff = align::diffAlignables(refComp[11],curComp[11]); align::moveAlignable(curComp[11], diff);
		edm::LogInfo("test") << "diffcheck: " << align::diffAlignables(refComp[11],curComp[11]);
		std::vector<Alignable*> refInnerStrip;
		std::vector<Alignable*> curInnerStrip;
		std::vector<AlgebraicVector> diffs;
		refInnerStrip.push_back(refComp[4]); refInnerStrip.push_back(refComp[5]); refInnerStrip.push_back(refComp[6]); refInnerStrip.push_back(refComp[7]);
		curInnerStrip.push_back(refComp[4]); curInnerStrip.push_back(refComp[5]); curInnerStrip.push_back(refComp[6]); curInnerStrip.push_back(refComp[7]);
		align::diffAlignables(refInnerStrip,curInnerStrip,diffs);
		for (unsigned int j = 0; j < refInnerStrip.size(); j++){
			align::moveAlignable(curInnerStrip[j],diffs[j]);
		}
		diffs.clear();
		align::diffAlignables(refInnerStrip,curInnerStrip,diffs);
		for (unsigned int j = 0; j < refInnerStrip.size(); j++){
			edm::LogInfo("test") << "diffcheck: " << diffs[j];
		}
	}
	//*/
	for (unsigned int i = 0; i < nComp; ++i) compareGeometries(refComp[i],curComp[i]);
}

void TrackerGeometryCompare::fillTree(Alignable *refAli, AlgebraicVector diff){

	//edm::LogInfo("compareGeometries") << "DIFF: " << diff;
	_id = refAli->geomDetId().rawId();
	_level = refAli->alignableObjectId();
	//need if ali has no mother
	if (refAli->mother()){
		_mid = refAli->mother()->geomDetId().rawId();
		_mlevel = refAli->mother()->alignableObjectId();
	}
	else{
		_mid = -1;
		_mlevel = -1;
	}
	DetId detid(_id);
	_sublevel = detid.subdetId();
	_xVal = refAli->globalPosition().x();
	_yVal = refAli->globalPosition().y();
	_zVal = refAli->globalPosition().z();
	//_rVal = sqrt(_xVal*_xVal + _yVal*_yVal);
	//_phiVal = atan(_yVal/_xVal);
	align::GlobalVector vec(_xVal,_yVal,_zVal);
	_rVal = vec.perp();
	_phiVal = vec.phi();
	align::RotationType rot = refAli->globalRotation();
	align::EulerAngles eulerAngles = align::toAngles(rot);
	_alphaVal = eulerAngles[0];
	_betaVal = eulerAngles[1];
	_gammaVal = eulerAngles[2];
	_dxVal = diff[0];
	_dyVal = diff[1];
	_dzVal = diff[2];
	//getting dR and dPhi
	align::GlobalVector vRef(_xVal,_yVal,_zVal);
	align::GlobalVector vCur(_xVal - _dxVal, _yVal - _dyVal, _zVal - _dzVal);
	_drVal = vCur.perp() - vRef.perp();
	_dphiVal = vCur.phi() - vRef.phi();
	_dalphaVal = diff[3];
	_dbetaVal = diff[4];
	_dgammaVal = diff[5];
	//Fill
	_alignTree->Fill();

}
