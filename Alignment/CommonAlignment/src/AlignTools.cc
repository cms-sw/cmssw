#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/CommonAlignment/interface/AlignTools.h"

#include <iostream>
#include <fstream>

//Finds the TR between two alignables - first alignable is reference
AlgebraicVector align::diffAlignables(Alignable* refAli, Alignable*curAli, std::string weightBy, bool weightById, std::string weightByIdFile){
	
	//check they are the same
	if (refAli->alignableObjectId() != curAli->alignableObjectId()){
		if (refAli->geomDetId().rawId() != curAli->geomDetId().rawId()){
			throw cms::Exception("Geometry Error")
			<< "[AlignTools] Error, Alignables do not match";
		}
	}
	
	//create points
	align::GlobalVectors refVs;
	align::GlobalVectors curVs;
	align::createPoints(&refVs, refAli, weightBy, weightById, weightByIdFile);
	align::createPoints(&curVs, curAli, weightBy, weightById, weightByIdFile);
	
	//redefine the set of points
	//find the translational difference
	align::GlobalVector theR = align::diffR(curVs,refVs);
	
	//CM difference (needed below in rotational transformation)
	align::GlobalVector pointsCM = align::centerOfMass(curVs);
	align::PositionType alignableCM = curAli->globalPosition();
	align::GlobalVector cmdiff(alignableCM.x()-pointsCM.x(), alignableCM.y()-pointsCM.y(), alignableCM.z()-pointsCM.z());
	
	
	//readjust points before finding rotation
	align::GlobalVector CMref = align::centerOfMass(refVs);
	align::GlobalVector CMcur = align::centerOfMass(curVs);
	for (unsigned int k = 0; k < refVs.size(); ++k){
		refVs[k] -= CMref;
		curVs[k] -= CMcur;
	}
	
	//find rotational difference
	align::RotationType rot = align::diffRot(curVs, refVs);
	align::EulerAngles theW = align::toAngles( rot );
	
	//adjust translational difference factoring in different rotational CM
	//needed because rotateInGlobalFrame is about CM of alignable, not points
	align::GlobalVector::BasicVectorType lpvgf = cmdiff.basicVector();
	align::GlobalVector moveV( rot.multiplyInverse(lpvgf) - lpvgf);
	align::GlobalVector theRprime(theR + moveV);
	
	AlgebraicVector deltaRW(6);
	deltaRW(1) = theRprime.x();
	deltaRW(2) = theRprime.y();
	deltaRW(3) = theRprime.z();
	deltaRW(4) = theW(1);
	deltaRW(5) = theW(2);
	deltaRW(6) = theW(3);
	
	refVs.clear();
	curVs.clear();
	
	return deltaRW;
	
}

//Moves the alignable by the AlgebraicVector
void align::moveAlignable(Alignable* ali, AlgebraicVector diff){
	
	GlobalVector dR(diff[0],diff[1],diff[2]);
	align::EulerAngles dOmega(3); dOmega[0] = diff[3]; dOmega[1] = diff[4]; dOmega[2] = diff[5];
	align::RotationType dRot = align::toMatrix(dOmega);
	ali->move(dR);
	ali->rotateInGlobalFrame(dRot);
	
}

//Creates the points which are used in diffAlignables
void align::createPoints(align::GlobalVectors* Vs, Alignable* ali, std::string weightBy, bool weightById, std::string weightByIdFile){
	
	
	const align::Alignables& comp = ali->components();
	unsigned int nComp = comp.size();
	for (unsigned int i = 0; i < nComp; ++i) align::createPoints(Vs, comp[i], weightBy, weightById, weightByIdFile);
	// double the weight for SS modules if weight by Det
	if ((ali->alignableObjectId() == align::AlignableDet)&&(weightBy == "Det")){
		for (unsigned int i = 0; i < nComp; ++i) align::createPoints(Vs, comp[i], weightBy, weightById, weightByIdFile);
	}
	
	//only create points for lowest hiearchical level
	if (ali->alignableObjectId() == align::AlignableDetUnit){
		//check if the raw id or the mother's raw id is on the list
		bool createPointsForDetUnit = true;
		if (weightById) createPointsForDetUnit = align::readModuleList( ali->id(), ali->mother()->id(), weightByIdFile);
		if (createPointsForDetUnit){
			//if no survey information, create local points
			if(!(ali->survey())){
				align::ErrorMatrix error;
				ali->setSurvey( new SurveyDet (ali->surface(), error*1e-6) );
			}
			const align::GlobalPoints& points = ali->surface().toGlobal(ali->survey()->localPoints());
			for (unsigned int j = 0; j < points.size(); ++j){
				align::GlobalVector dummy(points[j].x(),points[j].y(),points[j].z());
				Vs->push_back(dummy);
			}
		}
	}
	
}


bool align::readModuleList(unsigned int aliId, unsigned int motherId, std::string filename){
	
	bool foundId = false;
	
	std::ifstream inFile;
	inFile.open( filename.c_str() );
	int ctr = 0;
	while ( !inFile.eof() ){
		ctr++;
		unsigned int listId;
		inFile >> listId;
		inFile.ignore(256, '\n');
		
		if (listId == aliId){ foundId = true; break; }
		if (listId == motherId){ foundId = true; break; }
	}
	inFile.close();
	
	return foundId;
}



