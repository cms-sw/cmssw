/** \class DTRefitAndCombineReco4D
 *
 * Algo for reconstructing 4d segment in DT refitting the 2D phi SL hits and combining
 * the results with the theta view.
 *  
 * $Date: 2006/04/19 15:00:33 $
 * $Revision: 1.1 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

// #include "RecoLocalMuon/DTSegment/src/DTRecSegment2DBaseAlgo.h"
// #include "RecoLocalMuon/DTSegment/src/DTRecSegment2DAlgoFactory.h"

#include "RecoLocalMuon/DTSegment/src/DTSegmentUpdator.h"
#include "RecoLocalMuon/DTSegment/src/DTRefitAndCombineReco4D.h"

using namespace std;
using namespace edm;
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DPhi.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentCand.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "FWCore/Utilities/interface/Exception.h"

DTRefitAndCombineReco4D::DTRefitAndCombineReco4D(const ParameterSet& pset):
DTRecSegment4DBaseAlgo(pset), theAlgoName("DTRefitAndCombineReco4D"){

  // debug parameter
  debug = pset.getUntrackedParameter<bool>("debug");
  
  // the updator
  theUpdator = new DTSegmentUpdator(pset);

  // the max allowd chi^2 for the fit of th combination of two phi segments
  theMaxChi2forPhi = pset.getParameter<double>("MaxChi2forPhi");
  


//   // Get the concrete 2D-segments reconstruction algo from the factory
//   string theReco2DAlgoName = pset.getParameter<string>("Reco2DAlgoName");
//   cout << "the Reco2D AlgoName is " << theReco2DAlgoName << endl;
//   the2DAlgo = DTRecSegment2DAlgoFactory::get()->create(theReco2DAlgoName,
//                                                      pset.getParameter<ParameterSet>("Reco2DAlgoConfig"));
}

void DTRefitAndCombineReco4D::setES(const EventSetup& setup){
  setup.get<MuonGeometryRecord>().get(theDTGeometry);
  theUpdator->setES(setup);
  //  the2DAlgo->setES(setup);
}


  
OwnVector<DTRecSegment4D>
DTRefitAndCombineReco4D::reconstruct(const DTChamber* chamber,
					  const vector<DTRecSegment2D>& segments2DPhi1,
					  const vector<DTRecSegment2D>& segments2DTheta,
					  const vector<DTRecSegment2D>& segments2DPhi2){
  OwnVector<DTRecSegment4D> result;
  
  if (debug) cout << "Segments in " << chamber->id() << endl;

  vector<DTRecSegment2DPhi> resultPhi = refitSuperSegments(segments2DPhi1,segments2DPhi2);

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
    for (vector<DTRecSegment2DPhi>::const_iterator phi=resultPhi.begin();
         phi!=resultPhi.end(); ++phi) {
      
      if (hasZed) {

	// Create all the 4D-segment combining the Z view with the Phi one
	// loop over the Z segments
	for(vector<DTRecSegment2D>::const_iterator zed = segments2DTheta.begin();
	    zed != segments2DTheta.end(); ++zed){

	  //>> Important!!
	  DTSuperLayerId ZedSegSLId(zed->geographicalId().rawId());

	  const LocalPoint posZInCh  = chamber->toLocal( chamber->superLayer(ZedSegSLId)->toGlobal(zed->localPosition() )) ;
	  const LocalVector dirZInCh = chamber->toLocal( chamber->superLayer(ZedSegSLId)->toGlobal(zed->localDirection() )) ;
	  
	  DTRecSegment4D* newSeg = new DTRecSegment4D(*phi,*zed,posZInCh,dirZInCh);
	  //<<

          /// 4d segment: I have the pos along the wire => further update!
          theUpdator->update(newSeg);
          if (debug) cout << "Created a 4D seg " << endl;
	  result.push_back(newSeg);
        }
      } else {
        // Only phi
	// FIXME:implement this constructor
        DTRecSegment4D* newSeg = new DTRecSegment4D(*phi);
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
	//<<

        if (debug) cout << "Created a 4D segment using only the 2D Theta segment" << endl;
	result.push_back(newSeg);
      }
    }
  }
  
  return result;
}

vector<DTRecSegment2DPhi> DTRefitAndCombineReco4D::refitSuperSegments(const vector<DTRecSegment2D>& segments2DPhi1,
								      const vector<DTRecSegment2D>& segments2DPhi2){
  vector<DTRecSegment2DPhi> result;
  
  //double-loop over all the DTRecSegment2D in order to make all the possible pairs
  for(vector<DTRecSegment2D>::const_iterator segment2DPhi1 = segments2DPhi1.begin();
      segment2DPhi1 != segments2DPhi1.end(); ++segment2DPhi1){
    for(vector<DTRecSegment2D>::const_iterator segment2DPhi2 = segments2DPhi2.begin();
	segment2DPhi2 != segments2DPhi2.end(); ++segment2DPhi2){

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
      DTRecSegment2DPhi superPhi(chId,recHitsSeg2DPhi1); 
      
      // refit it!
      theUpdator->fit(&superPhi);
      
      // cut on the chi^2
      if (superPhi.chi2() > theMaxChi2forPhi)
	result.push_back(superPhi);
    }
  }
  // TODO clean the container!!!
  // maybe using the cleaner, previous a conversion from DTRecSegment2DPhi to DTSegmentCandidate
  return result;
}
