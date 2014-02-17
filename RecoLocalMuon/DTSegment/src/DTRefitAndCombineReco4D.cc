/** \class DTRefitAndCombineReco4D
 *
 * Algo for reconstructing 4d segment in DT refitting the 2D phi SL hits and combining
 * the results with the theta view.
 *  
 * $Date: 2006/04/28 15:21:52 $
 * $Revision: 1.5 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

#include "RecoLocalMuon/DTSegment/src/DTRefitAndCombineReco4D.h"

#include "RecoLocalMuon/DTSegment/src/DTSegmentUpdator.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentCand.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/DTRecHit/interface/DTChamberRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"

using namespace std;
using namespace edm;

DTRefitAndCombineReco4D::DTRefitAndCombineReco4D(const ParameterSet& pset):
DTRecSegment4DBaseAlgo(pset), theAlgoName("DTRefitAndCombineReco4D"){

  // debug parameter
  debug = pset.getUntrackedParameter<bool>("debug");
  
  // the updator
  theUpdator = new DTSegmentUpdator(pset);

  // the max allowd chi^2 for the fit of th combination of two phi segments
  theMaxChi2forPhi = pset.getParameter<double>("MaxChi2forPhi");
}

void DTRefitAndCombineReco4D::setES(const EventSetup& setup){
  setup.get<MuonGeometryRecord>().get(theDTGeometry);
  theUpdator->setES(setup);
  //  the2DAlgo->setES(setup);
}

void DTRefitAndCombineReco4D::setChamber(const DTChamberId &chId){
  // Set the chamber
  theChamber = theDTGeometry->chamber(chId); 
}

void
DTRefitAndCombineReco4D::setDTRecSegment2DContainer(Handle<DTRecSegment2DCollection> allHits) {
  theSegments2DPhi1.clear();
  theSegments2DTheta.clear();
  theSegments2DPhi2.clear();

  // Get the chamber
  //    const DTChamber *chamber = theDTGeometry->chamber(chId);
  
  const DTChamberId chId =  theChamber->id();
  
  //Extract the DTRecSegment2DCollection ranges for the three different SL
  DTRecSegment2DCollection::range rangePhi1   = allHits->get(DTSuperLayerId(chId,1));
  DTRecSegment2DCollection::range rangeTheta = allHits->get(DTSuperLayerId(chId,2));
  DTRecSegment2DCollection::range rangePhi2   = allHits->get(DTSuperLayerId(chId,3));
    
  // Fill the DTSLRecSegment2D containers for the three different SL
  vector<DTSLRecSegment2D> segments2DPhi1(rangePhi1.first,rangePhi1.second);
  vector<DTSLRecSegment2D> segments2DTheta(rangeTheta.first,rangeTheta.second);
  vector<DTSLRecSegment2D> segments2DPhi2(rangePhi2.first,rangePhi2.second);

  if(debug)
    cout << "Number of 2D-segments in the first  SL (Phi)" << segments2DPhi1.size() << endl
	 << "Number of 2D-segments in the second SL (Theta)" << segments2DTheta.size() << endl
	 << "Number of 2D-segments in the third  SL (Phi)" << segments2DPhi2.size() << endl;
    
  theSegments2DPhi1 = segments2DPhi1;
  theSegments2DTheta = segments2DTheta;
  theSegments2DPhi2 = segments2DPhi2;
}


  
OwnVector<DTRecSegment4D>
DTRefitAndCombineReco4D::reconstruct(){
  OwnVector<DTRecSegment4D> result;
  
  if (debug) cout << "Segments in " << theChamber->id() << endl;

  vector<DTChamberRecSegment2D> resultPhi = refitSuperSegments();

  if (debug) cout << "There are " << resultPhi.size() << " Phi cand" << endl;
  
  bool hasZed=false;

  // has this chamber the Z-superlayer?
  if (theSegments2DTheta.size()){
    hasZed = theSegments2DTheta.size()>0;
    if (debug) cout << "There are " << theSegments2DTheta.size() << " Theta cand" << endl;
  } else {
    if (debug) cout << "No Theta SL" << endl;
  }

  // Now I want to build the concrete DTRecSegment4D.
  if (resultPhi.size()) {
    for (vector<DTChamberRecSegment2D>::const_iterator phi=resultPhi.begin();
         phi!=resultPhi.end(); ++phi) {
      
      if (hasZed) {

	// Create all the 4D-segment combining the Z view with the Phi one
	// loop over the Z segments
	for(vector<DTSLRecSegment2D>::const_iterator zed = theSegments2DTheta.begin();
	    zed != theSegments2DTheta.end(); ++zed){

	  //>> Important!!
	  DTSuperLayerId ZedSegSLId(zed->geographicalId().rawId());

	  const LocalPoint posZInCh  = theChamber->toLocal( theChamber->superLayer(ZedSegSLId)->toGlobal(zed->localPosition() )) ;
	  const LocalVector dirZInCh = theChamber->toLocal( theChamber->superLayer(ZedSegSLId)->toGlobal(zed->localDirection() )) ;
	  
	  DTRecSegment4D* newSeg = new DTRecSegment4D(*phi,*zed,posZInCh,dirZInCh);
	  //<<

          /// 4d segment: I have the pos along the wire => further update!
          theUpdator->update(newSeg);
          if (debug) cout << "Created a 4D seg " << endl;
	  result.push_back(newSeg);
        }
      } else {
        // Only phi
        DTRecSegment4D* newSeg = new DTRecSegment4D(*phi);
        if (debug) cout << "Created a 4D segment using only the 2D Phi segment" << endl;
	result.push_back(newSeg);
      }
    }
  } else { 
    // DTRecSegment4D from zed projection only (unlikely, not so useful, but...)
    if (hasZed) {
      for(vector<DTSLRecSegment2D>::const_iterator zed = theSegments2DTheta.begin();
	  zed != theSegments2DTheta.end(); ++zed){
        
	// Important!!
	DTSuperLayerId ZedSegSLId(zed->geographicalId().rawId());

	const LocalPoint posZInCh  = theChamber->toLocal( theChamber->superLayer(ZedSegSLId)->toGlobal(zed->localPosition() )) ;
	const LocalVector dirZInCh = theChamber->toLocal( theChamber->superLayer(ZedSegSLId)->toGlobal(zed->localDirection() )) ;
	
	DTRecSegment4D* newSeg = new DTRecSegment4D( *zed,posZInCh,dirZInCh);
	//<<

        if (debug) cout << "Created a 4D segment using only the 2D Theta segment" << endl;
	result.push_back(newSeg);
      }
    }
  }
  
  return result;
}

vector<DTChamberRecSegment2D> DTRefitAndCombineReco4D::refitSuperSegments(){
  vector<DTChamberRecSegment2D> result;
  
  //double-loop over all the DTSLRecSegment2D in order to make all the possible pairs
  for(vector<DTSLRecSegment2D>::const_iterator segment2DPhi1 = theSegments2DPhi1.begin();
      segment2DPhi1 != theSegments2DPhi1.end(); ++segment2DPhi1){
    for(vector<DTSLRecSegment2D>::const_iterator segment2DPhi2 = theSegments2DPhi2.begin();
	segment2DPhi2 != theSegments2DPhi2.end(); ++segment2DPhi2){

      // check the id
      if(segment2DPhi1->chamberId() !=  segment2DPhi2->chamberId())
	throw cms::Exception("refitSuperSegments")
	  <<"he phi segments have different chamber id"<<std::endl;
      
      // create a super phi starting from 2 phi
      vector<DTRecHit1D> recHitsSeg2DPhi1 =  segment2DPhi1->specificRecHits();
      vector<DTRecHit1D> recHitsSeg2DPhi2 =  segment2DPhi2->specificRecHits();
      // copy the recHitsSeg2DPhi2 in the recHitsSeg2DPhi1 container
      copy(recHitsSeg2DPhi2.begin(),recHitsSeg2DPhi2.end(),back_inserter(recHitsSeg2DPhi1));
      
      const DTChamberId chId = segment2DPhi1->chamberId();

      // create the super phi
      DTChamberRecSegment2D superPhi(chId,recHitsSeg2DPhi1); 
      
      // refit it!
      theUpdator->fit(&superPhi);
      
      // cut on the chi^2
      if (superPhi.chi2() > theMaxChi2forPhi)
	result.push_back(superPhi);
    }
  }
  // TODO clean the container!!!
  // there are some possible repetition!
  // maybe using the cleaner, previous a conversion from DTChamberRecSegment2D to DTSegmentCandidate
  return result;
}
