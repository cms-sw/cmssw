#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CLHEP/Vector/RotationInterfaces.h" 
#include "CondFormats/Alignment/interface/AlignmentSorter.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorRcd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
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
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

//#include "Alignment/OfflineValidation/interface/ComparisonUtilities.h"
//#include "Alignment/CommonAlignment/interface/AlignTools.h"

//#include "Alignment/OfflineValidation/plugins/TrackerGeometryCompare.h"
#include "TrackerGeometryCompare.h"
#include "TFile.h" 
#include "CLHEP/Vector/ThreeVector.h"

// Database
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "Geometry/Records/interface/PGeometricDetRcd.h"

#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"

#include <iostream>
#include <fstream>

TrackerGeometryCompare::TrackerGeometryCompare(const edm::ParameterSet& cfg) :
  theSurveyIndex(0), _commonTrackerLevel(align::invalid), firstEvent_(true)
{
	
	//input is ROOT
	_inputFilename1 = cfg.getUntrackedParameter< std::string > ("inputROOTFile1");
	_inputFilename2 = cfg.getUntrackedParameter< std::string > ("inputROOTFile2");
	_inputTreename = cfg.getUntrackedParameter< std::string > ("treeName");
	
	//output file
	_filename = cfg.getUntrackedParameter< std::string > ("outputFile");
	
	_writeToDB = cfg.getUntrackedParameter< bool > ("writeToDB" );
	
	const std::vector<std::string>& levels = cfg.getUntrackedParameter< std::vector<std::string> > ("levels");
	
	_weightBy = cfg.getUntrackedParameter< std::string > ("weightBy");
	_setCommonTrackerSystem = cfg.getUntrackedParameter< std::string > ("setCommonTrackerSystem");
	_detIdFlag = cfg.getUntrackedParameter< bool > ("detIdFlag");
	_detIdFlagFile = cfg.getUntrackedParameter< std::string > ("detIdFlagFile");
	_weightById  = cfg.getUntrackedParameter< bool > ("weightById");
	_weightByIdFile = cfg.getUntrackedParameter< std::string > ("weightByIdFile");
	
	//setting the levels being used in the geometry comparator
	AlignableObjectId dummy;
	edm::LogInfo("TrackerGeometryCompare") << "levels: " << levels.size();
	for (unsigned int l = 0; l < levels.size(); ++l){
		theLevels.push_back( dummy.nameToType(levels[l]));
		edm::LogInfo("TrackerGeometryCompare") << "level: " << levels[l];
	}
	
		
	// if want to use, make id cut list
	if (_detIdFlag){
        ifstream fin;
        fin.open( _detIdFlagFile.c_str() );
        
        while (!fin.eof() && fin.good() ){
			
			uint32_t id;
			fin >> id;
			_detIdFlagVector.push_back(id);
        }
        fin.close();
	}		
	
	// turn weightByIdFile into weightByIdVector
	if (_weightById){
		std::ifstream inFile;
		inFile.open( _weightByIdFile.c_str() );
		int ctr = 0;
		while ( !inFile.eof() ){
			ctr++;
			unsigned int listId;
			inFile >> listId;
			inFile.ignore(256, '\n');
			
			_weightByIdVector.push_back( listId );
		}
		inFile.close();
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
	_alignTree->Branch("eta", &_etaVal, "eta/F");
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
	_alignTree->Branch("du", &_duVal, "du/F");
	_alignTree->Branch("dv", &_dvVal, "dv/F");
	_alignTree->Branch("dw", &_dwVal, "dw/F");
	_alignTree->Branch("da", &_daVal, "da/F");
	_alignTree->Branch("db", &_dbVal, "db/F");
	_alignTree->Branch("dg", &_dgVal, "dg/F");
	_alignTree->Branch("useDetId", &_useDetId, "useDetId/I");
	_alignTree->Branch("detDim", &_detDim, "detDim/I");	
	_alignTree->Branch("surW", &_surWidth, "surW/F");
	_alignTree->Branch("surL", &_surLength, "surL/F");
	_alignTree->Branch("surRot", &_surRot, "surRot[9]/D");
	_alignTree->Branch("identifiers", &_identifiers, "identifiers[6]/I");

	
}

void TrackerGeometryCompare::beginJob(){
  firstEvent_ = true;
}

void TrackerGeometryCompare::analyze(const edm::Event&, const edm::EventSetup& iSetup){

  if (firstEvent_) {

	//upload the ROOT geometries
	createROOTGeometry(iSetup);
	
	//set common tracker system first
	// if setting the tracker common system
	if (_setCommonTrackerSystem != "NONE"){
		setCommonTrackerSystem();
	}
	
	
	//compare the goemetries
	compareGeometries(referenceTracker,currentTracker);
	
	//write out ntuple
	//might be better to do within output module
	_theFile->cd();
	_alignTree->Write();
	_theFile->Close();
	
	
	if (_writeToDB){
		Alignments* myAlignments = currentTracker->alignments();
		AlignmentErrors* myAlignmentErrors = currentTracker->alignmentErrors();
		
		// 2. Store alignment[Error]s to DB
		edm::Service<cond::service::PoolDBOutputService> poolDbService;
		// Call service
		if( !poolDbService.isAvailable() ) // Die if not available
			throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
		
		poolDbService->writeOne<Alignments>(&(*myAlignments), poolDbService->beginOfTime(), "TrackerAlignmentRcd");
		poolDbService->writeOne<AlignmentErrors>(&(*myAlignmentErrors), poolDbService->beginOfTime(), "TrackerAlignmentErrorRcd");
		
	}		

	firstEvent_ = false;
  }
}

void TrackerGeometryCompare::createROOTGeometry(const edm::EventSetup& iSetup){
	
	int inputRawId1, inputRawId2;
	double inputX1, inputY1, inputZ1, inputX2, inputY2, inputZ2;
	double inputAlpha1, inputBeta1, inputGamma1, inputAlpha2, inputBeta2, inputGamma2;
		
	//declare alignments
	Alignments* alignments1 = new Alignments();
	AlignmentErrors* alignmentErrors1 = new AlignmentErrors();	
	if (_inputFilename1 != "IDEAL"){
		_inputRootFile1 = new TFile(_inputFilename1.c_str());
		TTree* _inputTree1 = (TTree*) _inputRootFile1->Get(_inputTreename.c_str());
		_inputTree1->SetBranchAddress("rawid", &inputRawId1);
		_inputTree1->SetBranchAddress("x", &inputX1);
		_inputTree1->SetBranchAddress("y", &inputY1);
		_inputTree1->SetBranchAddress("z", &inputZ1);
		_inputTree1->SetBranchAddress("alpha", &inputAlpha1);
		_inputTree1->SetBranchAddress("beta", &inputBeta1);
		_inputTree1->SetBranchAddress("gamma", &inputGamma1);
		
		int nEntries1 = _inputTree1->GetEntries();
		//fill alignments
		for (int i = 0; i < nEntries1; ++i){
			
			_inputTree1->GetEntry(i);
			CLHEP::Hep3Vector translation1(inputX1, inputY1, inputZ1);
			CLHEP::HepEulerAngles eulerangles1(inputAlpha1,inputBeta1,inputGamma1);
			uint32_t detid1 = inputRawId1;
			AlignTransform transform1(translation1, eulerangles1, detid1);
			alignments1->m_align.push_back(transform1);
			
			//dummy errors
			CLHEP::HepSymMatrix clhepSymMatrix(3,0);
			AlignTransformError transformError(clhepSymMatrix, detid1);
			alignmentErrors1->m_alignError.push_back(transformError);
		}		
		
		// to get the right order
		std::sort( alignments1->m_align.begin(), alignments1->m_align.end(), lessAlignmentDetId<AlignTransform>() );
		std::sort( alignmentErrors1->m_alignError.begin(), alignmentErrors1->m_alignError.end(), lessAlignmentDetId<AlignTransformError>() );
	}
	//------------------
	Alignments* alignments2 = new Alignments();
	AlignmentErrors* alignmentErrors2 = new AlignmentErrors();
	if (_inputFilename2 != "IDEAL"){	
		_inputRootFile2 = new TFile(_inputFilename2.c_str());
		TTree* _inputTree2 = (TTree*) _inputRootFile2->Get(_inputTreename.c_str());
		_inputTree2->SetBranchAddress("rawid", &inputRawId2);
		_inputTree2->SetBranchAddress("x", &inputX2);
		_inputTree2->SetBranchAddress("y", &inputY2);
		_inputTree2->SetBranchAddress("z", &inputZ2);
		_inputTree2->SetBranchAddress("alpha", &inputAlpha2);
		_inputTree2->SetBranchAddress("beta", &inputBeta2);
		_inputTree2->SetBranchAddress("gamma", &inputGamma2);
		
		int nEntries2 = _inputTree2->GetEntries();
		//fill alignments
		for (int i = 0; i < nEntries2; ++i){
			
			_inputTree2->GetEntry(i);
			CLHEP::Hep3Vector translation2(inputX2, inputY2, inputZ2);
			CLHEP::HepEulerAngles eulerangles2(inputAlpha2,inputBeta2,inputGamma2);
			uint32_t detid2 = inputRawId2;
			AlignTransform transform2(translation2, eulerangles2, detid2);
			alignments2->m_align.push_back(transform2);
			
			//dummy errors
			CLHEP::HepSymMatrix clhepSymMatrix(3,0);
			AlignTransformError transformError(clhepSymMatrix, detid2);
			alignmentErrors2->m_alignError.push_back(transformError);
		}			
		
		//to get the right order
		std::sort( alignments2->m_align.begin(), alignments2->m_align.end(), lessAlignmentDetId<AlignTransform>() );
		std::sort( alignmentErrors2->m_alignError.begin(), alignmentErrors2->m_alignError.end(), lessAlignmentDetId<AlignTransformError>() );
	}
	
	//accessing the initial geometry
	edm::ESTransientHandle<DDCompactView> cpv;
	iSetup.get<IdealGeometryRecord>().get(cpv);
	edm::ESHandle<GeometricDet> theGeometricDet;
	iSetup.get<IdealGeometryRecord>().get(theGeometricDet);
	TrackerGeomBuilderFromGeometricDet trackerBuilder;
	
	edm::ESHandle<Alignments> globalPositionRcd;
	iSetup.get<TrackerDigiGeometryRecord>().getRecord<GlobalPositionRcd>().get(globalPositionRcd);
	
	//reference tracker
	TrackerGeometry* theRefTracker = trackerBuilder.build(&*theGeometricDet); 
	if (_inputFilename1 != "IDEAL"){
		GeometryAligner aligner1;
		aligner1.applyAlignments<TrackerGeometry>( &(*theRefTracker), &(*alignments1), &(*alignmentErrors1),
												  align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Tracker)));
	}
	referenceTracker = new AlignableTracker(&(*theRefTracker));

	//currernt tracker
	TrackerGeometry* theCurTracker = trackerBuilder.build(&*theGeometricDet); 
	if (_inputFilename2 != "IDEAL"){
		GeometryAligner aligner2;
		aligner2.applyAlignments<TrackerGeometry>( &(*theCurTracker), &(*alignments2), &(*alignmentErrors2),
												  align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Tracker)));
	}
	currentTracker = new AlignableTracker(&(*theCurTracker));
	
	
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
	
	//another added level for difference between det and detunit
	//if ((refAli->alignableObjectId()==2)&&(nComp == 1)) useLevel = false;
	
	//coordinate matching, etc etc
	if (useLevel){
		//std::cout << "ali identifiers: " << refAli->id() << ", " << refAli->alignableObjectId() << std::endl;
		//std::cout << "diff pos" << (refAli->globalPosition() - curAli->globalPosition()) << std::endl;
		//std::cout <<"z";

		CLHEP::Hep3Vector Rtotal, Wtotal, lRtotal, lWtotal;

		Rtotal.set(0.,0.,0.); Wtotal.set(0.,0.,0.);
		lRtotal.set(0.,0.,0.); lWtotal.set(0.,0.,0.);
		for (int i = 0; i < 100; i++){
			AlgebraicVector diff = align::diffAlignables(refAli,curAli, _weightBy, _weightById, _weightByIdVector);
			CLHEP::Hep3Vector dR(diff[0],diff[1],diff[2]);
			Rtotal+=dR;
			CLHEP::Hep3Vector dW(diff[3],diff[4],diff[5]);
			CLHEP::HepRotation rot(Wtotal.unit(),Wtotal.mag());
			CLHEP::HepRotation drot(dW.unit(),dW.mag());
			rot*=drot;
			Wtotal.set(rot.axis().x()*rot.delta(), rot.axis().y()*rot.delta(), rot.axis().z()*rot.delta());
			// local coordinates
			lRtotal.set(diff[6],diff[7],diff[8]);
			lWtotal.set(diff[9],diff[10],diff[11]);
			//std::cout << "a";
			//if (refAli->alignableObjectId() == 1) std::cout << "DIFF: " << diff << std::endl;
			align::moveAlignable(curAli, diff);
			float tolerance = 1e-7;
			AlgebraicVector check = align::diffAlignables(refAli,curAli, _weightBy, _weightById, _weightByIdVector);
			align::GlobalVector checkR(check[0],check[1],check[2]);
			align::GlobalVector checkW(check[3],check[4],check[5]);
			DetId detid(refAli->id());
			if ((checkR.mag() > tolerance)||(checkW.mag() > tolerance)){
				edm::LogInfo("TrackerGeometryCompare") << "Tolerance Exceeded!(alObjId: " << refAli->alignableObjectId()
				<< ", rawId: " << refAli->geomDetId().rawId()
				<< ", subdetId: "<< detid.subdetId() << "): " << diff;
				throw cms::Exception("Tolerance in TrackerGeometryCompare exceeded");
			}
			else{
				break;
			}
		}
		
		AlgebraicVector TRtot(12);
		// global 
		TRtot(1) = Rtotal.x(); TRtot(2) = Rtotal.y(); TRtot(3) = Rtotal.z();
		TRtot(4) = Wtotal.x(); TRtot(5) = Wtotal.y(); TRtot(6) = Wtotal.z();
		// local
		TRtot(7) = lRtotal.x(); TRtot(8) = lRtotal.y(); TRtot(9) = lRtotal.z();
		TRtot(10) = lWtotal.x(); TRtot(11) = lWtotal.y(); TRtot(12) = lWtotal.z();
		fillTree(refAli, TRtot);
	}

	//another added level for difference between det and detunit
	for (unsigned int i = 0; i < nComp; ++i) 
	  compareGeometries(refComp[i],curComp[i]);	
}

void TrackerGeometryCompare::setCommonTrackerSystem(){

	edm::LogInfo("TrackerGeometryCompare") << "Setting Common Tracker System....";
	
	AlignableObjectId dummy;
	_commonTrackerLevel = dummy.nameToType(_setCommonTrackerSystem);
		
	diffCommonTrackerSystem(referenceTracker, currentTracker);
	
	align::EulerAngles dOmega(3); dOmega[0] = _TrackerCommonR.x() ; dOmega[1] = _TrackerCommonR.y(); dOmega[2] = _TrackerCommonR.z();
	align::RotationType rot = align::toMatrix( dOmega );
	align::GlobalVector theR = _TrackerCommonT;
	
	std::cout << "what we get from overlaying the pixels..." << theR << ", " << rot << std::endl;
	
	//transform to the Tracker System
	align::PositionType trackerCM = currentTracker->globalPosition();
	align::GlobalVector cmDiff( trackerCM.x()-_TrackerCommonCM.x(), trackerCM.y()-_TrackerCommonCM.y(), trackerCM.z()-_TrackerCommonCM.z() );
	
	std::cout << "Pixel CM: " << _TrackerCommonCM << ", tracker CM: " << trackerCM << std::endl;
	
	//adjust translational difference factoring in different rotational CM
	//needed because rotateInGlobalFrame is about CM of alignable, not Tracker
	align::GlobalVector::BasicVectorType lpvgf = cmDiff.basicVector();
	align::GlobalVector moveV( rot.multiplyInverse(lpvgf) - lpvgf);
	align::GlobalVector theRprime(theR + moveV);
	
	AlgebraicVector TrackerCommonTR(6);
	TrackerCommonTR(1) = theRprime.x(); TrackerCommonTR(2) = theRprime.y(); TrackerCommonTR(3) = theRprime.z();
	TrackerCommonTR(4) = _TrackerCommonR.x(); TrackerCommonTR(5) = _TrackerCommonR.y(); TrackerCommonTR(6) = _TrackerCommonR.z();
	
	std::cout << "and after the transformation: " << TrackerCommonTR << std::endl;
	
	align::moveAlignable(currentTracker, TrackerCommonTR );
	
}

void TrackerGeometryCompare::diffCommonTrackerSystem(Alignable *refAli, Alignable *curAli){
	
	const std::vector<Alignable*>& refComp = refAli->components();
	const std::vector<Alignable*>& curComp = curAli->components();
	
	unsigned int nComp = refComp.size();
	//only perform for designate levels
	bool useLevel = false;
	if (refAli->alignableObjectId() == _commonTrackerLevel) useLevel = true;
	
	//useLevel = false;
	if (useLevel){
		CLHEP::Hep3Vector Rtotal, Wtotal;
		Rtotal.set(0.,0.,0.); Wtotal.set(0.,0.,0.);
		
		AlgebraicVector diff = align::diffAlignables(refAli,curAli, _weightBy, _weightById, _weightByIdVector);
		CLHEP::Hep3Vector dR(diff[0],diff[1],diff[2]);
		Rtotal+=dR;
		CLHEP::Hep3Vector dW(diff[3],diff[4],diff[5]);
		CLHEP::HepRotation rot(Wtotal.unit(),Wtotal.mag());
		CLHEP::HepRotation drot(dW.unit(),dW.mag());
		rot*=drot;
		Wtotal.set(rot.axis().x()*rot.delta(), rot.axis().y()*rot.delta(), rot.axis().z()*rot.delta());
		/*
		 //std::cout << "a";
		 //if (refAli->alignableObjectId() == 1) std::cout << "DIFF: " << diff << std::endl;
		 align::moveAlignable(curAli, diff);
		 float tolerance = 1e-7;
		 AlgebraicVector check = align::diffAlignables(refAli,curAli, _weightBy, _weightById, _weightByIdVector);
		 align::GlobalVector checkR(check[0],check[1],check[2]);
		 align::GlobalVector checkW(check[3],check[4],check[5]);
		 DetId detid(refAli->id());
		 if ((checkR.mag() > tolerance)||(checkW.mag() > tolerance)){
		 edm::LogInfo("TrackerGeometryCompare") << "Tolerance Exceeded!(alObjId: " << refAli->alignableObjectId()
		 << ", rawId: " << refAli->geomDetId().rawId()
		 << ", subdetId: "<< detid.subdetId() << "): " << diff;
		 }
		 else{
		 break;
		 }
		 }
		 */
		
		//_TrackerCommonT.set(Rtotal.x(), Rtotal.y(), Rtotal.z());
		_TrackerCommonT = align::GlobalVector(Rtotal.x(), Rtotal.y(), Rtotal.z());
		_TrackerCommonR = align::GlobalVector(Wtotal.x(), Wtotal.y(), Wtotal.z());
		_TrackerCommonCM = curAli->globalPosition();
		//_TrackerCommonTR(1) = Rtotal.x(); _TrackerCommonTR(2) = Rtotal.y(); _TrackerCommonTR(3) = Rtotal.z();
		//_TrackerCommonTR(4) = Wtotal.x(); _TrackerCommonTR(5) = Wtotal.y(); _TrackerCommonTR(6) = Wtotal.z();
		
		
	}
	else{
		for (unsigned int i = 0; i < nComp; ++i) diffCommonTrackerSystem(refComp[i],curComp[i]);
	}
	
	
}

void TrackerGeometryCompare::fillTree(Alignable *refAli, AlgebraicVector diff){
	
	
	_id = refAli->id();
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
	fillIdentifiers( _sublevel, _id );
	_xVal = refAli->globalPosition().x();
	_yVal = refAli->globalPosition().y();
	_zVal = refAli->globalPosition().z();
	align::GlobalVector vec(_xVal,_yVal,_zVal);
	_rVal = vec.perp();
	_phiVal = vec.phi();
	_etaVal = vec.eta();
	align::RotationType rot = refAli->globalRotation();
	align::EulerAngles eulerAngles = align::toAngles(rot);
	_alphaVal = eulerAngles[0];
	_betaVal = eulerAngles[1];
	_gammaVal = eulerAngles[2];
	// global
	_dxVal = diff[0];
	_dyVal = diff[1];
	_dzVal = diff[2];
	// local
	_duVal = diff[6];
	_dvVal = diff[7];
	_dwVal = diff[8];
	//...TODO...
	align::GlobalVector g(_dxVal, _dyVal, _dzVal);
	//getting dR and dPhi
	align::GlobalVector vRef(_xVal,_yVal,_zVal);
	align::GlobalVector vCur(_xVal + _dxVal, _yVal + _dyVal, _zVal + _dzVal);
	_drVal = vCur.perp() - vRef.perp();
	_dphiVal = vCur.phi() - vRef.phi();
	// global
	_dalphaVal = diff[3];
	_dbetaVal = diff[4];
	_dgammaVal = diff[5];
	// local
	_daVal = diff[9];
	_dbVal = diff[10];
	_dgVal = diff[11];
	
	//detIdFlag
	if (refAli->alignableObjectId() == align::AlignableDetUnit){
		if (_detIdFlag){
			if ((passIdCut(refAli->id()))||(passIdCut(refAli->mother()->id()))){
				_useDetId = 1;
			}
			else{
				_useDetId = 0;
			}
		}
	}
	// det module dimension
	if (refAli->alignableObjectId() == align::AlignableDetUnit){
		if (refAli->mother()->alignableObjectId() != align::AlignableDet) _detDim = 1;
		else if (refAli->mother()->alignableObjectId() == align::AlignableDet) _detDim = 2;
	}
	else _detDim = 0;
	
	
	
	
	_surWidth = refAli->surface().width();
	_surLength = refAli->surface().length();
	align::RotationType rt = refAli->globalRotation();
	_surRot[0] = rt.xx(); _surRot[1] = rt.xy(); _surRot[2] = rt.xz();
	_surRot[3] = rt.yx(); _surRot[4] = rt.yy(); _surRot[5] = rt.yz();
	_surRot[6] = rt.zx(); _surRot[7] = rt.zy(); _surRot[8] = rt.zz();
	
	
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
		CLHEP::Hep3Vector clhepVector(pos.x(),pos.y(),pos.z());
		CLHEP::HepRotation clhepRotation( CLHEP::HepRep3x3(rot.xx(),rot.xy(),rot.xz(),rot.yx(),rot.yy(),rot.yz(),rot.zx(),rot.zy(),rot.zz()));
		AlignTransform transform(clhepVector, clhepRotation, (*k)->id());
		AlignTransformError transformError(CLHEP::HepSymMatrix(3,1), (*k)->id());
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

bool TrackerGeometryCompare::passIdCut( uint32_t id ){
	
	bool pass = false;
	int nEntries = _detIdFlagVector.size();
	
	for (int i = 0; i < nEntries; i++){
		if (_detIdFlagVector[i] == id) pass = true;
	}
	
	return pass;
	
}

void TrackerGeometryCompare::fillIdentifiers( int subdetlevel, int rawid ){
	
	
	switch( subdetlevel ){
			
		case 1:
		{
			PXBDetId pxbid( rawid );
			_identifiers[0] = pxbid.module();
			_identifiers[1] = pxbid.ladder();
			_identifiers[2] = pxbid.layer();
			_identifiers[3] = 999;
			_identifiers[4] = 999;
			_identifiers[5] = 999;
			break;
		}
		case 2:
		{
			PXFDetId pxfid( rawid );
			_identifiers[0] = pxfid.module();
			_identifiers[1] = pxfid.panel();
			_identifiers[2] = pxfid.blade();
			_identifiers[3] = pxfid.disk();
			_identifiers[4] = pxfid.side();
			_identifiers[5] = 999;
			break;
		}
		case 3:
		{
			TIBDetId tibid( rawid );
			_identifiers[0] = tibid.module();
			_identifiers[1] = tibid.string()[0];
			_identifiers[2] = tibid.string()[1];
			_identifiers[3] = tibid.string()[2];
			_identifiers[4] = tibid.layer();
			_identifiers[5] = 999;
			break;
		}
		case 4: 
		{
			TIDDetId tidid( rawid );
			_identifiers[0] = tidid.module()[0];
			_identifiers[1] = tidid.module()[1];
			_identifiers[2] = tidid.ring();
			_identifiers[3] = tidid.wheel();
			_identifiers[4] = tidid.side();
			_identifiers[5] = 999;
			break;
		}
		case 5: 
		{
			TOBDetId tobid( rawid );
			_identifiers[0] = tobid.module();
			_identifiers[1] = tobid.rod()[0];
			_identifiers[2] = tobid.rod()[1];
			_identifiers[3] = tobid.layer();
			_identifiers[4] = 999;
			_identifiers[5] = 999;
			break;
		}
		case 6: 
		{
			TECDetId tecid( rawid );
			_identifiers[0] = tecid.module();
			_identifiers[1] = tecid.ring();
			_identifiers[2] = tecid.petal()[0];
			_identifiers[3] = tecid.petal()[1];
			_identifiers[4] = tecid.wheel();
			_identifiers[5] = tecid.side();
			break;
		}
		default:
		{
			std::cout << "Error: bad subdetid!!" << std::endl;
			break;
		}
			
	}
}


DEFINE_FWK_MODULE(TrackerGeometryCompare);
