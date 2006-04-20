/** \file
 *
 * $Date: 2006/03/20 12:42:29 $
 * $Revision: 1.2 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 */

/* This Class Header */
#include "DataFormats/DTRecHit/interface/DTRecSegment2DPhi.h"

/* Collaborating Class Header */

/* C++ Headers */

/* ====================================================================== */

/// Constructor
DTRecSegment2DPhi::DTRecSegment2DPhi(const DetId& id) : DTRecSegment2D(id){

}

/// Destructor
DTRecSegment2DPhi::~DTRecSegment2DPhi() {
}


/// c'tor from hits
DTRecSegment2DPhi::DTRecSegment2DPhi(const DTChamberId& id, const std::vector<DTRecHit1D>& hits):
  DTRecSegment2D(id,hits){
}


/* Operations */ 


