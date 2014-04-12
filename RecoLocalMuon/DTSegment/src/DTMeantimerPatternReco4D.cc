/** \file
 *
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 */
#include "RecoLocalMuon/DTSegment/src/DTMeantimerPatternReco4D.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoLocalMuon/DTSegment/src/DTSegmentUpdator.h"
// For the 2D reco I use thei reconstructor!
#include "RecoLocalMuon/DTSegment/src/DTMeantimerPatternReco.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentCand.h"

#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/DTRecHit/interface/DTRangeMapAccessor.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"
#include "DataFormats/DTRecHit/interface/DTSLRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTChamberRecSegment2D.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"

using namespace std;
using namespace edm;

// TODO
// Throw an exception if a theta segment container is requested and in the event
// there isn't it. (Or launch a "lazy" reco on demand)

DTMeantimerPatternReco4D::DTMeantimerPatternReco4D(const ParameterSet& pset):
  DTRecSegment4DBaseAlgo(pset), theAlgoName("DTMeantimerPatternReco4D"){

    // debug parameter
    debug = pset.getUntrackedParameter<bool>("debug");

    //do you want the T0 correction?
    applyT0corr = pset.getParameter<bool>("performT0SegCorrection");

    computeT0corr = pset.existsAs<bool>("computeT0Seg") ?
      pset.getParameter<bool>("computeT0Seg") : true;

    // the updator
    theUpdator = new DTSegmentUpdator(pset);

    // the input type. 
    // If true the instructions in setDTRecSegment2DContainer will be schipped and the 
    // theta segment will be recomputed from the 1D rechits
    // If false the theta segment will be taken from the Event. Caveat: in this case the
    // event must contain the 2D segments!
    allDTRecHits = pset.getParameter<bool>("AllDTRecHits");

    // Get the concrete 2D-segments reconstruction algo from the factory
    // For the 2D reco I use this reconstructor!
    the2DAlgo = new DTMeantimerPatternReco(pset.getParameter<ParameterSet>("Reco2DAlgoConfig"));
}


DTMeantimerPatternReco4D::~DTMeantimerPatternReco4D(){
  delete the2DAlgo;
  delete theUpdator;
}

void DTMeantimerPatternReco4D::setES(const EventSetup& setup){
  setup.get<MuonGeometryRecord>().get(theDTGeometry);
  the2DAlgo->setES(setup);
  theUpdator->setES(setup);
}

void DTMeantimerPatternReco4D::setChamber(const DTChamberId &chId){
  // Set the chamber
  theChamber = theDTGeometry->chamber(chId); 
}

void DTMeantimerPatternReco4D::setDTRecHit1DContainer(Handle<DTRecHitCollection> all1DHits){
  theHitsFromPhi1.clear();
  theHitsFromPhi2.clear();
  theHitsFromTheta.clear();

  DTRecHitCollection::range rangeHitsFromPhi1 = 
    all1DHits->get(DTRangeMapAccessor::layersBySuperLayer( DTSuperLayerId(theChamber->id(),1) ) );
  DTRecHitCollection::range rangeHitsFromPhi2 = 
    all1DHits->get(DTRangeMapAccessor::layersBySuperLayer( DTSuperLayerId(theChamber->id(),3) ) );

  vector<DTRecHit1DPair> hitsFromPhi1(rangeHitsFromPhi1.first,rangeHitsFromPhi1.second);
  vector<DTRecHit1DPair> hitsFromPhi2(rangeHitsFromPhi2.first,rangeHitsFromPhi2.second);
  if(debug)
    cout<< "Number of DTRecHit1DPair in the SL 1 (Phi 1): " << hitsFromPhi1.size()<<endl
                                                               << "Number of DTRecHit1DPair in the SL 3 (Phi 2): " << hitsFromPhi2.size()<<endl;

  theHitsFromPhi1 = hitsFromPhi1;
  theHitsFromPhi2 = hitsFromPhi2;

  if(allDTRecHits){
    DTRecHitCollection::range rangeHitsFromTheta = 
      all1DHits->get(DTRangeMapAccessor::layersBySuperLayer( DTSuperLayerId(theChamber->id(),2) ) );

    vector<DTRecHit1DPair> hitsFromTheta(rangeHitsFromTheta.first,rangeHitsFromTheta.second);
    if(debug)
      cout<< "Number of DTRecHit1DPair in the SL 2 (Theta): " << hitsFromTheta.size()<<endl;
    theHitsFromTheta = hitsFromTheta;
  }

}

void DTMeantimerPatternReco4D::setDTRecSegment2DContainer(Handle<DTRecSegment2DCollection> all2DSegments){
  theSegments2DTheta.clear();

  if(!allDTRecHits){

    //Extract the DTRecSegment2DCollection range for the theta SL
    DTRecSegment2DCollection::range rangeTheta = 
      all2DSegments->get(DTSuperLayerId(theChamber->id(),2));

    // Fill the DTRecSegment2D container for the theta SL
    vector<DTSLRecSegment2D> segments2DTheta(rangeTheta.first,rangeTheta.second);

    if(debug)
      cout << "Number of 2D-segments in the second SL (Theta): " << segments2DTheta.size() << endl;
    theSegments2DTheta = segments2DTheta;
  }

}


OwnVector<DTRecSegment4D>
DTMeantimerPatternReco4D::reconstruct(){

  OwnVector<DTRecSegment4D> result;

  if (debug){ 
    cout << "Segments in " << theChamber->id() << endl;
    cout << "Reconstructing Phi segments"<<endl;
  }

  vector<std::shared_ptr<DTHitPairForFit>> pairPhiOwned;
  vector<DTSegmentCand*> resultPhi = buildPhiSuperSegmentsCandidates(pairPhiOwned);

  if (debug) cout << "There are " << resultPhi.size() << " Phi cand" << endl;

  if (allDTRecHits){
    // take the theta SL of this chamber
    const DTSuperLayer* sl = theChamber->superLayer(2);
    // sl points to 0 if the theta SL was not found
    if(sl){
      // reconstruct the theta segments
      if(debug) cout << "Reconstructing Theta segments"<<endl;
      OwnVector<DTSLRecSegment2D> thetaSegs = the2DAlgo->reconstruct(sl, theHitsFromTheta);
      vector<DTSLRecSegment2D> segments2DTheta(thetaSegs.begin(),thetaSegs.end());
      theSegments2DTheta = segments2DTheta;
    }
  }

  bool hasZed=false;

  // has this chamber the Z-superlayer?
  if (theSegments2DTheta.size()){
    hasZed = theSegments2DTheta.size()>0;
    if (debug) cout << "There are " << theSegments2DTheta.size() << " Theta cand" << endl;
  } else {
    if (debug) cout << "No Theta candidates." << endl;
  }

  // Now I want to build the concrete DTRecSegment4D.
  if(debug) cout << "Building the concrete DTRecSegment4D" << endl;
  if (resultPhi.size()) {
    for (vector<DTSegmentCand*>::const_iterator phi=resultPhi.begin();
         phi!=resultPhi.end(); ++phi) {

      std::auto_ptr<DTChamberRecSegment2D> superPhi(**phi);

      theUpdator->update(superPhi.get());
      if(debug) cout << "superPhi: " << *superPhi << endl;

      if (hasZed) {

        // Create all the 4D-segment combining the Z view with the Phi one
        // loop over the Z segments
        for(vector<DTSLRecSegment2D>::const_iterator zed = theSegments2DTheta.begin();
            zed != theSegments2DTheta.end(); ++zed){

          // Important!!
          DTSuperLayerId ZedSegSLId(zed->geographicalId().rawId());

	  // Put the theta segment poistion in its 3D place.
	  // note: (superPhi is in the CHAMBER local frame)
	  const DTSuperLayer* zSL = theChamber->superLayer(ZedSegSLId);

	  // FIXME: should rather extrapolate for Y!
	  LocalPoint zPos(zed->localPosition().x(), 
			 (zSL->toLocal(theChamber->toGlobal(superPhi->localPosition()))).y(),
			 0.);

          const LocalPoint posZInCh  = theChamber->toLocal( zSL->toGlobal(zPos));
	  // FIXME: zed->localDirection() is in 2D. Should add the phi direction in the orthogonal plane as well!!
          const LocalVector dirZInCh = theChamber->toLocal( zSL->toGlobal(zed->localDirection()));

          DTRecSegment4D* newSeg = new DTRecSegment4D(*superPhi,*zed,posZInCh,dirZInCh);

          /// 4d segment: I have the pos along the wire => further update!
          theUpdator->update(newSeg);
          if (debug) cout << "Created a 4D seg " << *newSeg << endl;

          //update the segment with the t0 and possibly vdrift correction
          if(!applyT0corr && computeT0corr) theUpdator->calculateT0corr(newSeg);
          if(applyT0corr) theUpdator->update(newSeg,true);

          result.push_back(newSeg);
        }
      } else {
        // Only phi
        DTRecSegment4D* newSeg = new DTRecSegment4D(*superPhi);

        if (debug) cout << "Created a 4D segment using only the 2D Phi segment" << endl;

        //update the segment with the t0 and possibly vdrift correction
        if(!applyT0corr && computeT0corr) theUpdator->calculateT0corr(newSeg);
        if(applyT0corr) theUpdator->update(newSeg,true);

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
        // <<

        if (debug) cout << "Created a 4D segment using only the 2D Theta segment" << endl;

        if(!applyT0corr && computeT0corr) theUpdator->calculateT0corr(newSeg);
        if(applyT0corr) theUpdator->update(newSeg,true);

        result.push_back(newSeg);
      }
    }
  }
  // finally delete the candidates!
  for (vector<DTSegmentCand*>::iterator phi=resultPhi.begin();
       phi!=resultPhi.end(); ++phi) delete *phi;

  return result;
}



vector<DTSegmentCand*> DTMeantimerPatternReco4D::buildPhiSuperSegmentsCandidates(vector<std::shared_ptr<DTHitPairForFit>> &pairPhiOwned){

  DTSuperLayerId slId;

  if(theHitsFromPhi1.size())
    slId = theHitsFromPhi1.front().wireId().superlayerId();
  else
    if(theHitsFromPhi2.size())
      slId = theHitsFromPhi2.front().wireId().superlayerId();
    else{
      if(debug) cout<<"DTMeantimerPatternReco4D::buildPhiSuperSegmentsCandidates: "
        <<"No Hits in the two Phi SL"<<endl;
      return vector<DTSegmentCand*>();
    }

  const DTSuperLayer *sl = theDTGeometry->superLayer(slId);

  vector<std::shared_ptr<DTHitPairForFit>> pairPhi1 = the2DAlgo->initHits(sl,theHitsFromPhi1);
  // same sl!! Since the fit will be in the sl phi 1!
  vector<std::shared_ptr<DTHitPairForFit>> pairPhi2 = the2DAlgo->initHits(sl,theHitsFromPhi2);
  // copy the pairPhi2 in the pairPhi1 vector 
  copy(pairPhi2.begin(),pairPhi2.end(),back_inserter(pairPhi1));

  pairPhiOwned.swap(pairPhi1);
  // Build the segment candidate
  return the2DAlgo->buildSegments(sl,pairPhiOwned);
}
