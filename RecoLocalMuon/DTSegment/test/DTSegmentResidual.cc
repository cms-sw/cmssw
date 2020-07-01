/******* \class DTSegmentResidual *******
 *
 *
 * \author : Stefano Lacaprara - INFN LNL <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 *********************************/

/* This Class Header */
#include "RecoLocalMuon/DTSegment/test/DTSegmentResidual.h"

/* Collaborating Class Header */
#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTChamberRecSegment2D.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

/* C++ Headers */
using namespace std;
#include <cmath>

/* ====================================================================== */

/* Constructor */
DTSegmentResidual::DTResidual::DTResidual(double v, double wd, double a, DTEnums::DTCellSide s)
    : value(v), wireDistance(wd), angle(a), side(s) {}

DTSegmentResidual::DTSegmentResidual(const DTRecSegment2D* seg, const DTSuperLayer* sl)
    : theSeg(seg), theCh(nullptr), theSL(sl) {}

DTSegmentResidual::DTSegmentResidual(const DTChamberRecSegment2D* seg, const DTChamber* ch)
    : theSeg(seg), theCh(ch), theSL(nullptr) {}

/* Operations */
void DTSegmentResidual::run() {
  vector<DTRecHit1D> hits = theSeg->specificRecHits();
  for (vector<DTRecHit1D>::const_iterator hit = hits.begin(); hit != hits.end(); ++hit) {
    // interpolate the segment to hit plane
    // get this layer position in SL frame
    const DTLayer* lay = theSL ? theSL->layer((*hit).wireId().layer()) : theCh->layer((*hit).wireId().layerId());
    LocalPoint layPosInSL = theSL ? theSL->toLocal(lay->position()) : theCh->toLocal(lay->position());

    LocalPoint posAtLay =
        theSeg->localPosition() + theSeg->localDirection() * (layPosInSL.z() / cos(theSeg->localDirection().theta()));
    posAtLay = lay->toLocal(theSL ? theSL->toGlobal(posAtLay) : theCh->toGlobal(posAtLay));

    double deltaX = (*hit).localPosition().x() - posAtLay.x();
    double angle = M_PI - theSeg->localDirection().theta();
    double wireDistance = (*hit).localPosition().x() - lay->specificTopology().wirePosition((*hit).wireId().wire());
    DTEnums::DTCellSide side = (*hit).lrSide();

    theResiduals.push_back(DTResidual(deltaX, wireDistance, angle, side));
  }
}
