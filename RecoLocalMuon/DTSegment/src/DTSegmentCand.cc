/** \file
 *
 * $Date: 2008/03/10 11:28:30 $
 * $Revision: 1.12 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 */

/* This Class Header */
#include "RecoLocalMuon/DTSegment/src/DTSegmentCand.h"

#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "DataFormats/DTRecHit/interface/DTSLRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTChamberRecSegment2D.h"

/* Collaborating Class Header */

/* C++ Headers */

/* ====================================================================== */
double DTSegmentCand::chi2max=20.; // to be tuned!!
unsigned int DTSegmentCand::nHitsMin=3; // to be tuned!!

/// Constructor
DTSegmentCand::DTSegmentCand(AssPointCont& hits,
                             const DTSuperLayer* sl) :
theSL(sl), theChi2(-1.) , theHits(hits){
}

DTSegmentCand::DTSegmentCand(AssPointCont hits,
                             LocalPoint& position,
                             LocalVector& direction,
                             double chi2,
                             AlgebraicSymMatrix covMat,
                             const DTSuperLayer* sl) :
theSL(sl), thePosition(position), theDirection(direction), theChi2(chi2),
  theCovMatrix( covMat), theHits(hits) {
}

/// Destructor
DTSegmentCand::~DTSegmentCand() {
}

/* Operations */ 
bool DTSegmentCand::operator==(const DTSegmentCand& seg){
  static const double epsilon=0.00001;
  if (nHits()!=seg.nHits()) return false;
  if (fabs(chi2()-seg.chi2())>epsilon) return false;
  if (fabs(position().x()-seg.position().x())>epsilon ||
      fabs(position().y()-seg.position().y())>epsilon ||
      fabs(position().z()-seg.position().z())>epsilon) return false;
  if (fabs(direction().x()-seg.direction().x())>epsilon ||
      fabs(direction().y()-seg.direction().y())>epsilon ||
      fabs(direction().z()-seg.direction().z())>epsilon) return false;
  return true;
}

bool DTSegmentCand::operator<(const DTSegmentCand& seg){
  if (nHits()==seg.nHits()) return (chi2()>seg.chi2());
  return (nHits()<seg.nHits());
}

void DTSegmentCand::add(DTHitPairForFit* hit, DTEnums::DTCellSide code) {
  theHits.insert(AssPoint(hit,code));
}

void DTSegmentCand::removeHit(AssPoint badHit) {
  theHits.erase(badHit);
}

int DTSegmentCand::nSharedHitPairs(const DTSegmentCand& seg) const{
  int result=0;
  AssPointCont hitsCont = seg.hits();

  for (AssPointCont::const_iterator hit=theHits.begin(); 
       hit!=theHits.end() ; ++hit) {
    for (AssPointCont::const_iterator hit2=hitsCont.begin();
         hit2!=hitsCont.end() ; ++hit2) {
      //  if(result) return result ; // TODO, uncomm this line or move it in another func
      if ((*(*hit).first)==(*(*hit2).first)) {
        ++result;
        continue;
      }
    }
  }
  return result;
}

DTSegmentCand::AssPointCont
DTSegmentCand::conflictingHitPairs(const DTSegmentCand& seg) const{
  AssPointCont result;
  AssPointCont hitCont = seg.hits();
  
//  if (nSharedHitPairs(seg)==0) return result;

  for (AssPointCont::const_iterator hit=theHits.begin(); 
       hit!=theHits.end() ; ++hit) {
    for (AssPointCont::const_iterator hit2 = hitCont.begin();
         hit2!=hitCont.end() ; ++hit2) {
      if ((*(*hit).first)==(*(*hit2).first) &&
          (*hit).second != (*hit2).second) {
        result.insert(*hit);
        continue;
      }
    }
  }
  return result;
}

bool DTSegmentCand::good() const {
  return nHits()>=nHitsMin && chi2()/NDOF() < chi2max ;
}

int DTSegmentCand::nLayers() const {
  // TODO
  return 0;
}

DTSegmentCand::operator DTSLRecSegment2D*() const{
  
  LocalPoint seg2Dposition = position();
  LocalVector seg2DDirection = direction();
  double seg2DChi2 = chi2();
  AlgebraicSymMatrix seg2DCovMatrix = covMatrix();
  
  std::vector<DTRecHit1D> hits1D;
  for(DTSegmentCand::AssPointCont::iterator assHit=theHits.begin();
      assHit!=theHits.end(); ++assHit) {

    GlobalPoint hitGlobalPos =
      theSL->toGlobal( (*assHit).first->localPosition((*assHit).second) );

    LocalPoint hitPosInLayer = 
      theSL->layer( (*assHit).first->id().layerId() )->toLocal(hitGlobalPos);

    DTRecHit1D hit( ((*assHit).first)->id(),
		    (*assHit).second,
		    ((*assHit).first)->digiTime(),
		    hitPosInLayer,
		    ((*assHit).first)->localPositionError() );
    hits1D.push_back(hit);
  }
  
  return new DTSLRecSegment2D(theSL->id(),
			    seg2Dposition,seg2DDirection,seg2DCovMatrix,
			    seg2DChi2,hits1D);
}

DTSegmentCand::operator DTChamberRecSegment2D*() const{
  
  // input position and direction are in sl frame, while must be stored in
  // chamber one: so I have to extrapolate the position (along the direction) to
  // the chamber reference plane.

  LocalPoint posInCh = theSL->chamber()->toLocal(theSL->toGlobal( position() ));
  LocalVector dirInCh= theSL->chamber()->toLocal(theSL->toGlobal( direction() ));
  
  LocalPoint pos=posInCh + dirInCh * posInCh.z()/cos(dirInCh.theta());
  
  double seg2DChi2 = chi2();
  AlgebraicSymMatrix seg2DCovMatrix = covMatrix();
  
  std::vector<DTRecHit1D> hits1D;
  for(DTSegmentCand::AssPointCont::iterator assHit=theHits.begin();
      assHit!=theHits.end(); ++assHit) {

    GlobalPoint hitGlobalPos =
      theSL->toGlobal( (*assHit).first->localPosition((*assHit).second) );
    
    LocalPoint hitPosInLayer = 
      theSL->chamber()
      ->superLayer((*assHit).first->id().superlayerId())
      ->layer( (*assHit).first->id().layerId() )->toLocal(hitGlobalPos);
    
    DTRecHit1D hit( ((*assHit).first)->id(),
		    (*assHit).second,
		    ((*assHit).first)->digiTime(),
		    hitPosInLayer,
		    ((*assHit).first)->localPositionError() );
    hits1D.push_back(hit);
  }
  
  return new DTChamberRecSegment2D(theSL->chamber()->id(),
				   pos,dirInCh,seg2DCovMatrix,
				   seg2DChi2,hits1D);
  
  // chamber and Phi SLs' frame are oriented in the same way, only a transaltion,
  // so the covariance matrix is the same!
}


bool DTSegmentCand::AssPointLessZ::operator()(const AssPoint& pt1, 
                                              const AssPoint& pt2) const {
  return *(pt1.first) < *(pt2.first);
}

std::ostream& operator<<(std::ostream& out, const DTSegmentCand& seg) {
  out <<  " chi2/nHits: " << seg.chi2() << "/" << seg.nHits() ;
  return out;
}

std::ostream& operator<<(std::ostream& out, const DTSegmentCand::AssPoint& hit) {
  // out << "Hits " << (hit.first)->localPosition(DTEnums::Left) <<
  //     " " << hit.second  << " Lay " << (hit.first)->layerNumber() << endl;
  return out;
}
