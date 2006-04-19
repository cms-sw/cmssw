/** \file
 *
 * $Date:  $
 * $Revision: $
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
    hasZed = (segments2DTheta.size()>0);
    if (debug) cout << "There are " << segments2DTheta.size() << " Theta cand" << endl;
  } else {
    if (debug) cout << "No Theta SL" << endl;
  }

  // Now I want to build the concrete MuBarSegment.
  if (resultPhi.size()) {
    for (vector<DTSegmentCand*>::iterator phi=resultPhi.begin();
         phi!=resultPhi.end(); ++phi) {
      
      //FIXME, check the converter and change its name
      DTRecSegment2DPhi* superPhi = (*phi)->convert(chamber);
      
      theUpdator->update(superPhi);
      
      /*
      // << start
      if (hasZed) {
        // TODO must create a DTRecSegment4D out of RecHit, not DTRecSegment2D
        vector<RecHit> zeds=SLtheta->recHits();
        for (vector<RecHit>::iterator zed=zeds.begin();
             zed!=zeds.end(); ++zed) {
          MuBarSegment* newSeg = new MuBarSegment(superPhi,*zed,chamber);

          /// 4d segment: I have the pos along the wire => further update!
          theUpdator.update(newSeg);
          if (coutUV.infoOut) cout << "Created a 4D seg " << endl;
          chRecDet->add(RecHit(newSeg));
        }
      } else {
        // Only phi
        //cout << "adding MuBarSegment(superPhi)" << endl;
        MuBarSegment* newSeg = new MuBarSegment(superPhi,chamber);
        if (coutUV.infoOut) cout << "Created a 2D seg (Phi)" << endl;
        chRecDet->add(RecHit(newSeg));
      }
    }
  } else { 
    // MuBarSegment from zed projection only (unlikely, not so useful, but...
    if (hasZed) {
      vector<RecHit> zeds=SLtheta->recHits();
      for (vector<RecHit>::iterator zed=zeds.begin();
           zed!=zeds.end(); ++zed) {
        //cout << "adding MuBarSegment(*zed))" << endl;
        MuBarSegment* newSeg= new MuBarSegment(*zed,chamber);
        if (coutUV.infoOut) cout << "Created a 2D seg (Zed)" << endl;
        chRecDet->add(RecHit(newSeg));
      }
    }
  }
  // finally delete the candidates!
  for (vector<MuBarSegmentCand*>::iterator phi=resultPhi.begin();
  phi!=resultPhi.end(); ++phi) delete *phi;

  //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  //stop
  */
    } // to be removed when uncomm
  }
  return result;
}
