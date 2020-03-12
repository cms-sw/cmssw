/******* \class DTSegmentExtendedCand *******
 *
 * Description:
 *  
 *  detailed description
 *
 * \author : Stefano Lacaprara - INFN LNL <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 *********************************/

/* This Class Header */
#include "RecoLocalMuon/DTSegment/src/DTSegmentExtendedCand.h"

/* Collaborating Class Header */

/* C++ Headers */
#include <iostream>
using namespace std;

/* ====================================================================== */

/* Constructor */

/* Destructor */

/* Operations */
bool DTSegmentExtendedCand::isCompatible(const DTSegmentExtendedCand::DTSLRecClusterForFit& clus) {
  LocalPoint posAtSL = position() + direction() * (clus.pos.z() - position().z()) / cos(direction().theta());
  // cout << "pos :" << clus.pos << " posAtSL " << posAtSL << endl;
  static constexpr float errScaleFact = 10.;
  static constexpr float minError = 25.;  // (cm)
  // cout << "clus.err.xx() " << clus.err << endl;
  return std::abs((posAtSL - clus.pos).x()) < max(errScaleFact * sqrt(clus.err.xx()), minError);
}

unsigned int DTSegmentExtendedCand::nHits() const { return DTSegmentCand::nHits() + theClus.size(); }

bool DTSegmentExtendedCand::good() const {
  if (superLayer()->id().superLayer() == 2)
    return DTSegmentCand::nHits() >= nHitsMin && chi2() / NDOF() < chi2max * 2.;
  return DTSegmentCand::nHits() >= nHitsMin && chi2() / NDOF() < chi2max;
}
