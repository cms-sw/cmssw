/** \file
 *
 * $Date: 2009/11/27 11:59:48 $
 * $Revision: 1.9 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 */

/* This Class Header */
#include "RecoLocalMuon/DTSegment/src/DTHitPairForFit.h"

/* Collaborating Class Header */
#include "FWCore/Utilities/interface/Exception.h"

/* C++ Headers */
#include <iostream>
using namespace std;

/* ====================================================================== */

/// Constructor
DTHitPairForFit::DTHitPairForFit(const DTRecHit1DPair& pair,
                                 const DTSuperLayer& sl,
                                 const edm::ESHandle<DTGeometry>& dtGeom) {

  theWireId = pair.wireId();
  theDigiTime = pair.digiTime();
  
  const DTLayer* layer = dtGeom->layer(theWireId.layerId());

  // transform the Local position in Layer-rf in a SL local position
  theLeftPos =
    sl.toLocal(layer->toGlobal(pair.componentRecHit(DTEnums::Left)->localPosition()));
  theRightPos =
    sl.toLocal(layer->toGlobal(pair.componentRecHit(DTEnums::Right)->localPosition()));

  // TODO how do I transform an error from local to global?
  theError = pair.componentRecHit(DTEnums::Left)->localPositionError();
  // theError =
  //   layer->surface().toLocal(sl.surface().toGlobal(pair.componentRecHit(DTEnums::Left)->localPositionError()));
  
}

/// Destructor
DTHitPairForFit::~DTHitPairForFit() {
}

/* Operations */ 
LocalPoint DTHitPairForFit::localPosition(DTEnums::DTCellSide s) const {
  if (s==DTEnums::Left) return theLeftPos;
  else if (s==DTEnums::Right) return theRightPos;
  else{ 
    throw cms::Exception("DTHitPairForFit")<<" localPosition called with undef LR code"<<endl;
    return LocalPoint();
  }
}

pair<bool,bool> 
DTHitPairForFit::isCompatible(const LocalPoint& posIni,
                              const LocalVector& dirIni) const {


    pair<bool,bool> ret;
    LocalPoint segPosAtZLeft  = posIni+dirIni*(theLeftPos.z() -posIni.z())/dirIni.z();
    LocalPoint segPosAtZRight = posIni+dirIni*(theRightPos.z()-posIni.z())/dirIni.z();
    float dxLeft  = fabs(theLeftPos.x() - segPosAtZLeft.x());
    float dxRight = fabs(theRightPos.x() - segPosAtZRight.x());
    float exx = sqrt(theError.xx());
    // if both below 3 sigma, return both
    // if both at 10 sigma or above, return none
    // if one is below N sigma and one above, for 10>=N>=3, match only that one, otherwise none
    if (std::max(dxLeft, dxRight) < 3*exx) {
        ret = make_pair(true,true);
    } else if (std::min(dxLeft, dxRight) >= 10*exx) {
        ret = make_pair(false,false);
    } else {
        float sigmasL = floorf(dxLeft/exx), sigmasR = floorf(dxRight/exx);
        ret.first  = ( sigmasL < sigmasR );
        ret.second = ( sigmasR < sigmasL );
    } 
    return ret;
}

bool DTHitPairForFit::operator<(const DTHitPairForFit& hit) const {
  //SL if same layer use x() for strict ordering
  if (id()==hit.id()) 
    return (theLeftPos.x() < hit.localPosition(DTEnums::Left).x());
  return (id() < hit.id());
}

bool DTHitPairForFit::operator==(const DTHitPairForFit& hit) const {
  return  (id() == hit.id() && fabs(digiTime() - hit.digiTime()) < 0.1 );
}

ostream& operator<<(ostream& out, const DTHitPairForFit& hit) {
  out << hit.leftPos() << " " << hit.rightPos() ;
  return out;
}
