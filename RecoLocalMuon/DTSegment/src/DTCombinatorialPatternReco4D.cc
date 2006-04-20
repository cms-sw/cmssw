/** \file
 *
 * $Date: 2006/04/19 17:39:53 $
 * $Revision: 1.2 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 */

#include "RecoLocalMuon/DTSegment/src/DTRecSegment2DBaseAlgo.h"
#include "RecoLocalMuon/DTSegment/src/DTRecSegment2DAlgoFactory.h"

#include "RecoLocalMuon/DTSegment/src/DTSegmentUpdator.h"
#include "RecoLocalMuon/DTSegment/src/DTCombinatorialPatternReco4D.h"

using namespace std;
using namespace edm;
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DPhi.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentCand.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

DTCombinatorialPatternReco4D::DTCombinatorialPatternReco4D(const ParameterSet& pset):
  DTRecSegment4DBaseAlgo(pset), theAlgoName("DTCombinatorialPatternReco4D"){

  // debug parameter
  debug = pset.getUntrackedParameter<bool>("debug");
  
  // the updator
  theUpdator = new DTSegmentUpdator(pset);
  
  // Get the concrete 2D-segments reconstruction algo from the factory
  string theReco2DAlgoName = pset.getParameter<string>("Reco2DAlgoName");
  cout << "the Reco2D AlgoName is " << theReco2DAlgoName << endl;
  the2DAlgo = DTRecSegment2DAlgoFactory::get()->create(theReco2DAlgoName,
						       pset.getParameter<ParameterSet>("Reco2DAlgoConfig"));
}

void DTCombinatorialPatternReco4D::setES(const EventSetup& setup){
  setup.get<MuonGeometryRecord>().get(theDTGeometry);
  the2DAlgo->setES(setup);
}


  
OwnVector<DTRecSegment4D>
DTCombinatorialPatternReco4D::reconstruct(const DTChamber* chamber,
					  const vector<DTRecSegment2D>& segments2DPhi1,
					  const vector<DTRecSegment2D>& segments2DTheta,
					  const vector<DTRecSegment2D>& segments2DPhi2){
  OwnVector<DTRecSegment4D> result;
  
  if (debug) cout << "Segments in " << chamber->id() << endl;

  
  // FIXME!! It isn't in the abstract interface!!
  vector<DTSegmentCand*> resultPhi;
  //  vector<DTSegmentCand*> resultPhi = the2DAlgo->buildPhiSuperSegments(segments2DPhi1,segments2DPhi2);
  
  if (debug) cout << "There are " << resultPhi.size() << " Phi cand" << endl;
  
  bool hasZed=false;

  // has this chamber the Z-superlayer?
  if (segments2DTheta.size()){
    hasZed = segments2DTheta.size()>0;
    if (debug) cout << "There are " << segments2DTheta.size() << " Theta cand" << endl;
  } else {
    if (debug) cout << "No Theta SL" << endl;
  }

  // Now I want to build the concrete DTRecSegment4D.
  if (resultPhi.size()) {
    for (vector<DTSegmentCand*>::const_iterator phi=resultPhi.begin();
         phi!=resultPhi.end(); ++phi) {
      
      //FIXME, check the converter and change its name
      DTRecSegment2DPhi* superPhi = (*phi)->convert(chamber);
      
      theUpdator->update(superPhi);
      
      
      // << start
      if (hasZed) {

	// Create all the 4D-segment combining the Z view with the Phi one
	// loop over the Z segments
	for(vector<DTRecSegment2D>::const_iterator zed = segments2DTheta.begin();
	    zed != segments2DTheta.end(); ++zed){
	  
	  // Important!!
	  DTSuperLayerId ZedSegSLId(zed->geographicalId().rawId());
	  
	  const LocalPoint posZInCh  = chamber->toLocal( chamber->superLayer(ZedSegSLId)->toGlobal(zed->localPosition() )) ;
	  const LocalVector dirZInCh = chamber->toLocal( chamber->superLayer(ZedSegSLId)->toGlobal(zed->localDirection() )) ;
	
          DTRecSegment4D* newSeg = new DTRecSegment4D(*superPhi,*zed,posZInCh,dirZInCh);
	  //<<
  
          /// 4d segment: I have the pos along the wire => further update!
          theUpdator->update(newSeg);
          if (debug) cout << "Created a 4D seg " << endl;
	  result.push_back(newSeg);
        }
      } else {
        // Only phi
        DTRecSegment4D* newSeg = new DTRecSegment4D(*superPhi);
	
        if (debug) cout << "Created a 4D segment using only the 2D Phi segment" << endl;
	result.push_back(newSeg);
      }
    }
  } else { 
    // DTRecSegment4D from zed projection only (unlikely, not so useful, but...)
    if (hasZed) {
      for(vector<DTRecSegment2D>::const_iterator zed = segments2DTheta.begin();
	  zed != segments2DTheta.end(); ++zed){
        
	// Important!!
	DTSuperLayerId ZedSegSLId(zed->geographicalId().rawId());
	  
	const LocalPoint posZInCh  = chamber->toLocal( chamber->superLayer(ZedSegSLId)->toGlobal(zed->localPosition() )) ;
	const LocalVector dirZInCh = chamber->toLocal( chamber->superLayer(ZedSegSLId)->toGlobal(zed->localDirection() )) ;
	
        DTRecSegment4D* newSeg = new DTRecSegment4D( *zed,posZInCh,dirZInCh);
	// <<
	
        if (debug) cout << "Created a 4D segment using only the 2D Theta segment" << endl;
	result.push_back(newSeg);
      }
    }
  }
  // finally delete the candidates!
  for (vector<DTSegmentCand*>::iterator phi=resultPhi.begin();
       phi!=resultPhi.end(); ++phi) delete *phi;
  
  return result;
}



// vector<DTSegmentCand*> DTCombinatorialPatternReco4D::buildPhiSuperSegments(const vector<DTRecSegment2D>& segments2DPhi1,
// 									   const vector<DTRecSegment2D>& segments2DPhi2){
// }
