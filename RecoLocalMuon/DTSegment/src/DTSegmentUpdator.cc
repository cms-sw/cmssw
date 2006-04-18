/** \file
 *
 * $Date: 2006/04/13 15:43:06 $
 * $Revision: 1.5 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 */

/* This Class Header */
#include "RecoLocalMuon/DTSegment/src/DTSegmentUpdator.h"

/* Collaborating Class Header */
#include "RecoLocalMuon/DTRecHit/interface/DTRecHitAlgoFactory.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentCand.h"
#include "RecoLocalMuon/DTSegment/interface/DTLinearFit.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

/* C++ Headers */
#include <string>
using namespace std;
using namespace edm;

/* ====================================================================== */

/// Constructor
DTSegmentUpdator::DTSegmentUpdator(const ParameterSet& config) :
theFitter(new DTLinearFit()) {  
  string theAlgoName = config.getParameter<string>("recAlgo");
  theAlgo = DTRecHitAlgoFactory::get()->create(theAlgoName, config.getParameter<ParameterSet>("recAlgoConfig"));
}

/// Destructor
DTSegmentUpdator::~DTSegmentUpdator() {
  delete theFitter;
}

/* Operations */ 
// bool DTSegmentUpdator::update(DTRecSegment* seg) {
// }

void DTSegmentUpdator::setES(const edm::EventSetup& setup){
  setup.get<MuonGeometryRecord>().get(theGeom);
  theAlgo->setES(setup);
}

void DTSegmentUpdator::update(DTRecSegment2D* seg)  {
  updateHits(seg);
  fit(seg);
}

bool DTSegmentUpdator::fit(DTSegmentCand* seg) {
  if (!seg->good()) return false;

  vector<float> x;
  vector<float> y;
  vector<float> sigy;

  DTSegmentCand::AssPointCont hits=seg->hits();
  for (DTSegmentCand::AssPointCont::const_iterator iter=hits.begin();
       iter!=hits.end(); ++iter) {
    LocalPoint pos = (*iter).first->localPosition((*iter).second);
    x.push_back(pos.z()); 
    y.push_back(pos.x());
    sigy.push_back(sqrt((*iter).first->localPositionError().xx()));
  }

  LocalPoint pos;
  LocalVector dir;
  AlgebraicSymMatrix covMat(2);
  double chi2=0;
  fit(x,y,sigy,pos,dir,covMat,chi2);

  seg->setPosition(pos);
  seg->setDirection(dir);

  //cout << "pos " << segPosition<< endl;
  //cout << "dir " << segDirection<< endl;

  seg->setCovMatrix(covMat);
  // cout << "Mat " << covMat << endl;

  seg->setChi2(chi2);
  return true;
}

void DTSegmentUpdator::fit(DTRecSegment2D* seg) {

  vector<float> x;
  vector<float> y;
  vector<float> sigy;

  vector<DTRecHit1D> hits=seg->specificRecHits();
  for (vector<DTRecHit1D>::const_iterator hit=hits.begin();
       hit!=hits.end(); ++hit) {

    // TODO move from hit frame to SL frame
    // RB: is my transformation right? 

    // I have to get the hits position (the hit is in the layer rf) in SL frame...
    GlobalPoint glbPos = ( theGeom->idToDet(hit->geographicalId()) )->toGlobal(hit->localPosition());
    LocalPoint pos = ( theGeom->idToDet(seg->geographicalId()) )->toLocal(glbPos);

    x.push_back(pos.z()); 
    y.push_back(pos.x());

    // Get local error in SL frame
    //RB,FIXME: is it right in this way? 
    ErrorFrameTransformer tran;
    GlobalError glbErr =
      tran.transform( hit->localPositionError(),(theGeom->idToDet(hit->geographicalId()))->surface());
    // RB, I prefer:
    //  tran.transform( hit->localPositionError(),(theGeom->layer(hit->geographicalId()))->surface());
    LocalError slErr =
      tran.transform( glbErr, (theGeom->idToDet(seg->geographicalId()))->surface());
    // RB, I prefer:
    // tran.transform( glbErr, (theGeom->superLayer(seg->geographicalId()))->surface());
    
    sigy.push_back(sqrt(slErr.xx()));
  }
  
  LocalPoint pos;
  LocalVector dir;
  AlgebraicSymMatrix covMat(2);
  double chi2=0;
  fit(x,y,sigy,pos,dir,covMat,chi2);

  seg->setPosition(pos);

  seg->setDirection(dir);

  //cout << "pos " << segPosition<< endl;
  //cout << "dir " << segDirection<< endl;

  seg->setCovMatrix(covMat);
  // cout << "Mat " << mat << endl;

  seg->setChi2(chi2);
}

void DTSegmentUpdator::fit(const vector<float>& x,
                           const vector<float>& y, 
                           const vector<float>& sigy,
                           LocalPoint& pos,
                           LocalVector& dir,
                           AlgebraicSymMatrix& covMatrix,
                           double& chi2) {
  float slope,intercept,covss,covii,covsi;
  // do the fit
  theFitter->fit(x,y,x.size(),sigy,slope,intercept,covss,covii,covsi);
  // cout << "slope " << slope << endl;
  // cout << "intercept " << intercept << endl;

  // intercept is the x() in chamber frame when the segment cross the chamber
  // plane (at z()=0), the y() is not measured, so let's put the center of the
  // chamber.
  pos = LocalPoint(intercept,0.,0.);

  //  slope is dx()/dz(), while dy()/dz() is by definition 0, finally I want the
  //  segment to point outward, so opposite to local z
  dir = LocalVector(-slope,0.,-1.).unit();

  covMatrix = AlgebraicSymMatrix(2);
  covMatrix[0][0] = covss; // this is var(dy/dz)
  covMatrix[1][1] = covii; // this is var(y)
  covMatrix[1][0] = covsi; // this is cov(dy/dz,y)

  /* Calculate chi2. */
  chi2 = 0.;
  for(unsigned int i=0; i<x.size() ; ++i) {
    double resid= y[i] - (intercept + slope*x[i]);
    chi2 += (resid/sigy[i])*(resid/sigy[i]);
  }
}

void DTSegmentUpdator::updateHits(DTRecSegment2D* seg) {

  //FIXME, RB: put a dynamic cast
  GlobalPoint pos = (theGeom->idToDet(seg->geographicalId()))->surface().toGlobal(seg->localPosition());
  GlobalVector dir = (theGeom->idToDet(seg->geographicalId()))->surface().toGlobal(seg->localDirection());
  updateHits(seg, pos, dir);
}

void DTSegmentUpdator::updateHits(DTRecSegment2D* seg,
                                  GlobalPoint gpos,
                                  GlobalVector gdir,
                                  int step) {
  /// define impact angle
  LocalPoint segPos=theGeom->idToDet(seg->geographicalId())->surface().toLocal(gpos);
  LocalVector segDir=theGeom->idToDet(seg->geographicalId())->surface().toLocal(gdir);
  LocalPoint segPosAtLayer=segPos+segDir*segPos.z()/cos(segDir.theta());
  const float angle = atan(segDir.x()/-segDir.z());


  // it is not necessary to have DTRecHit1D* to modify the obj in the container
  // but I have to be carefully, since I cannot make a copy before the iteration!

  vector<DTRecHit1D> toBeUpdatedRecHits = seg->specificRecHits();
  vector<DTRecHit1D> updatedRecHits;

  for (vector<DTRecHit1D>::iterator hit= toBeUpdatedRecHits.begin(); 
       hit!=toBeUpdatedRecHits.end(); ++hit) {
    // LocalPoint leftPoint;
    // LocalPoint rightPoint;
    // LocalError error;
    // TODO needed?
    // TODO How do I get a DTDigi (or a drift time) from a DTRecHit1D???
    // build a DTDigi on the fly just to please the algo
    // DetUnit* du=(*hit)->det().detUnits().front();
    // DTDigi::ChannelType channel = (*hit)->channels().front();
    // int countTDC=du->readout().channelAdc(channel);
    // DTDigi digi(channel,countTDC);
    //    DTDigi digi;

    const DTLayer* layer = theGeom->layer( hit->wireId().layerId() );

    DTRecHit1D newHit1D=(*hit);

    bool ok=true;

    if (step==2) {
      ok = theAlgo->compute(layer,
			    (*hit),
			    angle,
			    newHit1D);
    } else if (step==3) {
      
      LocalPoint hitPos(hit->localPosition().x(),+segPosAtLayer.y(),0.);
      
      //GlobalPoint gpos=theGeom.idToDet(hit->id())->toGlobal(hitPos);
      
      //FIXME,RB: is it right?
      GlobalPoint gpos= theGeom->layer( hit->wireId().layerId() )->surface().toGlobal(hitPos);

      ok = theAlgo->compute(layer,
			    (*hit),
			    angle,gpos,
			    newHit1D);
    } else {
      throw cms::Exception("DTSegmentUpdator")<<" updateHits called with wrong step"<<endl;
    }

    if (ok) {

      updatedRecHits.push_back(newHit1D);
      //      (*hit) = newHit1D;

      // RB,FIXME. was
      //      if (hit->lrSide()==DTEnums::Left ) hit->setPosition(leftPoint);
      // else if (hit->lrSide()==DTEnums::Right ) hit->setPosition(rightPoint);
      //   hit->setError(error);
      // is my line right?

    } else {
      LogError("DTSegmentUpdator")<<"DTSegmentUpdator::updateHits failed update" << endl;
      throw cms::Exception("DTSegmentUpdator")<<"updateHits failed update"<<endl;
    }
  }
  seg->update(updatedRecHits);
}
