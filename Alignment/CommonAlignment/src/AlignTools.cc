#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Alignment/CommonAlignment/interface/AlignTools.h"
#include "DataFormats/DetId/interface/DetId.h"

//Finds the TR between two alignables - first alignable is reference
AlgebraicVector align::diffAlignables(Alignable* refAli, Alignable*curAli){

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
	align::createPoints(&refVs, refAli);
	align::createPoints(&curVs, curAli);

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

//Finds the TR between two sets of alignables - first alignable is reference,
void align::diffAlignables(std::vector<Alignable*>& refAlis, std::vector<Alignable*>& curAlis, std::vector<AlgebraicVector>& diffs){

	align::GlobalVectors refVs;
	align::GlobalVectors curVs;
	
	unsigned int nAlis = refAlis.size();
	for (unsigned int j = 0; j < nAlis; ++j){
		
		//check they are the same
		if (refAlis[j]->alignableObjectId() != curAlis[j]->alignableObjectId()){
			if (refAlis[j]->geomDetId().rawId() != curAlis[j]->geomDetId().rawId()){
				throw cms::Exception("Geometry Error")
					<< "[AlignTools] Error, Alignables do not match";
			}
		}
		//create points
		align::createPoints(&refVs, refAlis[j]);
		align::createPoints(&curVs, curAlis[j]);

	}
	//find translational difference
	align::GlobalVector theR = align::diffR(curVs, refVs);
	
	//readjust points before finding rotation
	align::GlobalVector CMref = align::centerOfMass(refVs);
	align::GlobalVector CMcur = align::centerOfMass(curVs);
	for (unsigned int k = 0; k < refVs.size(); ++k){
		refVs[k] -= CMref;
		curVs[k] -= CMcur;
	}


	align::RotationType rot = align::diffRot(curVs, refVs);
	

	//have to change theR for each daughter because of frame change
	//in the future may turn into small utility method
	align::GlobalVector theCM = align::centerOfMass(curVs);
	for (unsigned int j = 0; j < nAlis; ++j){
		align::PositionType pt = curAlis[j]->globalPosition();
		align::GlobalVector vec(pt.x(),pt.y(),pt.z());
		align::GlobalVector lpv = vec - theCM;
		align::GlobalVector::BasicVectorType lpvgf = lpv.basicVector();
		align::GlobalVector moveV( rot.multiplyInverse(lpvgf) - lpvgf);
		align::GlobalVector theRprime(theR + moveV);

		align::EulerAngles theW = align::toAngles( rot );
		
		AlgebraicVector deltaRW(6);
		deltaRW(1) = theRprime.x();
		deltaRW(2) = theRprime.y();
		deltaRW(3) = theRprime.z();
		deltaRW(4) = theW(1);
		deltaRW(5) = theW(2);
		deltaRW(6) = theW(3);

		diffs.push_back(deltaRW);
	}
		
	refVs.clear();
	curVs.clear();

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
void align::createPoints(align::GlobalVectors* Vs, Alignable* ali){

	const std::vector<Alignable*> comp = ali->components();
	unsigned int nComp = comp.size();
	for (unsigned int i = 0; i < nComp; ++i) createPoints(Vs, comp[i]);

	//only create points for lowest hiearchical level
	if (ali->alignableObjectId() == AlignableObjectId::AlignableDetUnit){
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

	



