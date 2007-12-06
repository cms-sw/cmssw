#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CLHEP/Vector/RotationInterfaces.h" 
#include "CondFormats/Alignment/interface/AlignmentSorter.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h" 
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignTools.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Alignment/OfflineValidation/interface/TrackerGeometryCompare.h"
#include "TFile.h" 
#include "CLHEP/Vector/ThreeVector.h"

TrackerGeometryCompare::TrackerGeometryCompare(const edm::ParameterSet& cfg)
{
	//for root input
	//type of inputs being used
	_inputFileType = cfg.getUntrackedParameter< std::string > ("inputFileType");

	//if input is ROOT
	_inputFilename1 = cfg.getUntrackedParameter< std::string > ("inputROOTFile1");
	_inputFilename2 = cfg.getUntrackedParameter< std::string > ("inputROOTFile2");
	_inputTreename = cfg.getUntrackedParameter< std::string > ("treeName");
	
	//if input is DB
	_inputType = cfg.getUntrackedParameter< std::string > ("inputDBType");

	//output file
	_filename = cfg.getUntrackedParameter< std::string > ("outputFile");

	const std::vector<std::string>& levels = cfg.getUntrackedParameter< std::vector<std::string> > ("levels");
	//const std::vector<int>& subdets = cfg.getUntrackedParameter< std::vector<int> > ("subdets");



	//setting the levels being used in the geometry comparator
	AlignableObjectId dummy;
	edm::LogInfo("TrakcerGeomertyCompare") << "levels: " << levels.size();
	for (unsigned int l = 0; l < levels.size(); ++l){
		theLevels.push_back( dummy.nameToType(levels[l]));
		edm::LogInfo("TrakcerGeomertyCompare") << "level: " << levels[l];
	}
	//for (unsigned int l = 0; l < levels.size(); ++l){
	//	theSubDets.push_back( subdets[l] );
	//}
	


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

	//create the geometries
	if (_inputFileType == "poolora"){
		createDBGeometry(iSetup);
	}
	else if (_inputFileType == "root"){
		createROOTGeometry(iSetup);
	}
	else{
		throw cms::Exception("Bad File Type") << "Bad Input File Type";
	}

	//compare the goemetries
	compareGeometries(referenceTracker,currentTracker);
				

	//write out ntuple
	//might be better to do within output module
	_theFile->cd();
	_alignTree->Write();
	_theFile->Close();
	
}

void TrackerGeometryCompare::createDBGeometry(const edm::EventSetup& iSetup){

	edm::LogInfo("DBGeom") << "Creating Geoemtries from database...";
	
	//accessing the initial geometry
	edm::ESHandle<GeometricDet> theGeometricDet;
	iSetup.get<IdealGeometryRecord>().get(theGeometricDet);
	TrackerGeomBuilderFromGeometricDet trackerBuilder;
	//reference tracker
	TrackerGeometry* theRefTracker = trackerBuilder.build(&*theGeometricDet); 
	referenceTracker = new AlignableTracker(&(*theRefTracker));
	//dummy tracker
	dummyTracker = new AlignableTracker(&(*theRefTracker));
	

	//currernt tracker
	TrackerGeometry* theCurTracker = trackerBuilder.build(&*theGeometricDet); 
	
	if (_inputType == "tracker"){
		edm::ESHandle<Alignments> alignments;
		edm::ESHandle<AlignmentErrors> alignmentErrors;
		
		iSetup.get<TrackerAlignmentRcd>().get(alignments);
		iSetup.get<TrackerAlignmentErrorRcd>().get(alignmentErrors);

		//apply the latest alignments
		GeometryAligner aligner;
		aligner.applyAlignments<TrackerGeometry>( &(*theCurTracker), &(*alignments), &(*alignmentErrors));
		currentTracker = new AlignableTracker(&(*theCurTracker));
		
	}
	if (_inputType == "survey"){
		edm::ESHandle<SurveyValues> valuesHandle;
		edm::ESHandle<SurveyErrors> errorsHandle;

		iSetup.get<TrackerSurveyRcd>().get(valuesHandle);
		iSetup.get<TrackerSurveyErrorRcd>().get(errorsHandle);

		//add the survey info
		theSurveyIndex = 0;
		theSurveyValues = &*valuesHandle;
		theSurveyErrors = &*errorsHandle;
		addSurveyInfo(dummyTracker);
		
		//convert survey into alignments
		Alignments* alignVals = new Alignments();
		AlignmentErrors* alignErrors = new AlignmentErrors();
		surveyToTracker(&(*dummyTracker), alignVals, alignErrors); 
		
		//apply the survey alignments
		GeometryAligner aligner;
		aligner.applyAlignments<TrackerGeometry>( &(*theCurTracker), alignVals, alignErrors);
		currentTracker = new AlignableTracker(&(*theCurTracker));
			
	}
	
}

void TrackerGeometryCompare::createROOTGeometry(const edm::EventSetup& iSetup){

	_inputRootFile1 = new TFile(_inputFilename1.c_str());
	TTree* _inputTree1 = (TTree*) _inputRootFile1->Get(_inputTreename.c_str());
	_inputRootFile2 = new TFile(_inputFilename2.c_str());
	TTree* _inputTree2 = (TTree*) _inputRootFile2->Get(_inputTreename.c_str());
	
	

	//loop through the ROOT file
	int nEntries1 = _inputTree1->GetEntries();
	int nEntries2 = _inputTree1->GetEntries();
	if (nEntries1 != nEntries2) edm::LogInfo("creatROOT") << "nEntries incompatible: " << nEntries1 << ",  " << nEntries2;
	//read the ROOT file
	//uint32_t inputRawId;
	int inputRawId1, inputRawId2;
	double inputX1, inputY1, inputZ1, inputX2, inputY2, inputZ2;
	double inputAlpha1, inputBeta1, inputGamma1, inputAlpha2, inputBeta2, inputGamma2;
	_inputTree1->SetBranchAddress("rawid", &inputRawId1);
	_inputTree1->SetBranchAddress("x", &inputX1);
	_inputTree1->SetBranchAddress("y", &inputY1);
	_inputTree1->SetBranchAddress("z", &inputZ1);
	_inputTree1->SetBranchAddress("alpha", &inputAlpha1);
	_inputTree1->SetBranchAddress("beta", &inputBeta1);
	_inputTree1->SetBranchAddress("gamma", &inputGamma1);
	_inputTree2->SetBranchAddress("rawid", &inputRawId2);
	_inputTree2->SetBranchAddress("x", &inputX2);
	_inputTree2->SetBranchAddress("y", &inputY2);
	_inputTree2->SetBranchAddress("z", &inputZ2);
	_inputTree2->SetBranchAddress("alpha", &inputAlpha2);
	_inputTree2->SetBranchAddress("beta", &inputBeta2);
	_inputTree2->SetBranchAddress("gamma", &inputGamma2);

	//declare alignments
	Alignments* alignments1 = new Alignments();
	AlignmentErrors* alignmentErrors1 = new AlignmentErrors();
	Alignments* alignments2 = new Alignments();
	AlignmentErrors* alignmentErrors2 = new AlignmentErrors();

	//fill alignments
	for (int i = 0; i < nEntries1; ++i){
		
		_inputTree1->GetEntry(i);
		Hep3Vector translation1(inputX1, inputY1, inputZ1);
		HepEulerAngles eulerangles1(inputAlpha1,inputBeta1,inputGamma1);
		uint32_t detid1 = inputRawId1;
		AlignTransform transform1(translation1, eulerangles1, detid1);
		alignments1->m_align.push_back(transform1);

		_inputTree2->GetEntry(i);
		Hep3Vector translation2(inputX2, inputY2, inputZ2);
		HepEulerAngles eulerangles2(inputAlpha2,inputBeta2,inputGamma2);
		uint32_t detid2 = inputRawId2;
		AlignTransform transform2(translation2, eulerangles2, detid2);
		alignments2->m_align.push_back(transform2);
		
		//dummy errors
		HepSymMatrix clhepSymMatrix(3,0);
		AlignTransformError transformError(clhepSymMatrix, detid1);
		alignmentErrors1->m_alignError.push_back(transformError);
		alignmentErrors2->m_alignError.push_back(transformError);
	}
	//to get the right order
	std::sort( alignments1->m_align.begin(), alignments1->m_align.end(), lessAlignmentDetId<AlignTransform>() );
	std::sort( alignmentErrors1->m_alignError.begin(), alignmentErrors1->m_alignError.end(), lessAlignmentDetId<AlignTransformError>() );
	std::sort( alignments2->m_align.begin(), alignments2->m_align.end(), lessAlignmentDetId<AlignTransform>() );
	std::sort( alignmentErrors2->m_alignError.begin(), alignmentErrors2->m_alignError.end(), lessAlignmentDetId<AlignTransformError>() );


	//accessing the initial geometry
	edm::ESHandle<DDCompactView> cpv;
	iSetup.get<IdealGeometryRecord>().get(cpv);
	edm::ESHandle<GeometricDet> theGeometricDet;
	iSetup.get<IdealGeometryRecord>().get(theGeometricDet);
	TrackerGeomBuilderFromGeometricDet trackerBuilder;

	//reference tracker
	TrackerGeometry* theRefTracker = trackerBuilder.build(&*theGeometricDet); 
	GeometryAligner aligner1;
	aligner1.applyAlignments<TrackerGeometry>( &(*theRefTracker), &(*alignments1), &(*alignmentErrors1));
	referenceTracker = new AlignableTracker(&(*theRefTracker));
	//currernt tracker
	TrackerGeometry* theCurTracker = trackerBuilder.build(&*theGeometricDet); 
	GeometryAligner aligner2;
	aligner2.applyAlignments<TrackerGeometry>( &(*theCurTracker), &(*alignments2), &(*alignmentErrors2));
	currentTracker = new AlignableTracker(&(*theCurTracker));
	

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

	if (useLevel){
		AlgebraicVector diff = align::diffAlignables(refAli,curAli);
		//if (refAli->alignableObjectId() == 1) std::cout << "DIFF: " << diff << std::endl;
		fillTree(refAli, diff);
		align::moveAlignable(curAli, diff);
		float tolerance = 1e-7;
		diff = align::diffAlignables(refAli,curAli);
		align::GlobalVector check1(diff[0],diff[1],diff[2]);
		align::GlobalVector check2(diff[3],diff[4],diff[5]);
		DetId detid(refAli->id());
		if ((check1.mag() > tolerance)||(check2.mag() > tolerance)){
			edm::LogInfo("CopareGeoms") << "Tolerance Exceeded!(alObjId: " << refAli->alignableObjectId()
																	<< ", rawId: " << refAli->geomDetId().rawId()
																	<< ", subdetId: "<< detid.subdetId() << "): " << diff;
		}

	}
	

	for (unsigned int i = 0; i < nComp; ++i) compareGeometries(refComp[i],curComp[i]);

}

void TrackerGeometryCompare::fillTree(Alignable *refAli, AlgebraicVector diff){


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

void TrackerGeometryCompare::surveyToTracker(AlignableTracker* ali, Alignments* alignVals, AlignmentErrors* alignErrors){

	//getting the right alignables for the alignment record
	std::vector<Alignable*> detPB = ali->pixelHalfBarrelGeomDets();
	std::vector<Alignable*> detPEC = ali->pixelEndcapGeomDets();
	std::vector<Alignable*> detTIB = ali->innerBarrelGeomDets();
	std::vector<Alignable*> detTID = ali->TIDGeomDets();
	std::vector<Alignable*> detTOB = ali->outerBarrelGeomDets();
	std::vector<Alignable*> detTEC = ali->endcapGeomDets();

	std::vector<Alignable*> allGeomDets;
	std::copy(detPB.begin(), detPB.end(), std::back_inserter(allGeomDets));
	std::copy(detPEC.begin(), detPEC.end(), std::back_inserter(allGeomDets));
	std::copy(detTIB.begin(), detTIB.end(), std::back_inserter(allGeomDets));
	std::copy(detTID.begin(), detTID.end(), std::back_inserter(allGeomDets));
	std::copy(detTOB.begin(), detTOB.end(), std::back_inserter(allGeomDets));
	std::copy(detTEC.begin(), detTEC.end(), std::back_inserter(allGeomDets));
	
	std::vector<Alignable*> rcdAlis;
	for (std::vector<Alignable*>::iterator i = allGeomDets.begin(); i!= allGeomDets.end(); i++){
		if ((*i)->components().size() == 1){
			rcdAlis.push_back((*i));
		}
		else if ((*i)->components().size() > 1){
			rcdAlis.push_back((*i));
			std::vector<Alignable*> comp = (*i)->components();
			for (std::vector<Alignable*>::iterator j = comp.begin(); j != comp.end(); j++){
				rcdAlis.push_back((*j));
			}
		}
	}

	//turning them into alignments
	for(std::vector<Alignable*>::iterator k = rcdAlis.begin(); k != rcdAlis.end(); k++){

		const SurveyDet* surveyInfo = (*k)->survey();
		align::PositionType pos(surveyInfo->position());
		align::RotationType rot(surveyInfo->rotation());
		Hep3Vector clhepVector(pos.x(),pos.y(),pos.z());
		HepRotation clhepRotation( HepRep3x3(rot.xx(),rot.xy(),rot.xz(),rot.yx(),rot.yy(),rot.yz(),rot.zx(),rot.zy(),rot.zz()));
		AlignTransform transform(clhepVector, clhepRotation, (*k)->id());
		AlignTransformError transformError(HepSymMatrix(3,1), (*k)->id());
		alignVals->m_align.push_back(transform);
		alignErrors->m_alignError.push_back(transformError);
	}

	//to get the right order
	std::sort( alignVals->m_align.begin(), alignVals->m_align.end(), lessAlignmentDetId<AlignTransform>() );
	std::sort( alignErrors->m_alignError.begin(), alignErrors->m_alignError.end(), lessAlignmentDetId<AlignTransformError>() );
	
}

void TrackerGeometryCompare::addSurveyInfo(Alignable* ali){

	const std::vector<Alignable*>& comp = ali->components();

  unsigned int nComp = comp.size();

  for (unsigned int i = 0; i < nComp; ++i) addSurveyInfo(comp[i]);

  const SurveyError& error = theSurveyErrors->m_surveyErrors[theSurveyIndex];

  if ( ali->geomDetId().rawId() != error.rawId() ||
       ali->alignableObjectId() != error.structureType() )
  {
    throw cms::Exception("DatabaseError")
      << "Error reading survey info from DB. Mismatched id!";
  }

  const CLHEP::Hep3Vector&  pos = theSurveyValues->m_align[theSurveyIndex].translation();
  const CLHEP::HepRotation& rot = theSurveyValues->m_align[theSurveyIndex].rotation();

  AlignableSurface surf( align::PositionType( pos.x(), pos.y(), pos.z() ),
			 align::RotationType( rot.xx(), rot.xy(), rot.xz(),
					      rot.yx(), rot.yy(), rot.yz(),
					      rot.zx(), rot.zy(), rot.zz() ) );

  surf.setWidth( ali->surface().width() );
  surf.setLength( ali->surface().length() );

  ali->setSurvey( new SurveyDet( surf, error.matrix() ) );

	++theSurveyIndex;
	
}

