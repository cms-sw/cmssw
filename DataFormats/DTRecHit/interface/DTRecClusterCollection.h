#ifndef DTRECHIT_DTRECCLUSTERCOLLECTION_H
#define DTRECHIT_DTRECCLUSTERCOLLECTION_H

/** \class DTRecClusterCollection
 *
 * Description:
 *  
 *  detailed description
 *
 * \author : Stefano Lacaprara - INFN LNL <stefano.lacaprara@pd.infn.it>
 * $date   : 17/04/2008 15:14:33 CEST $
 *
 * Modification:
 *
 */

/* Base Class Headers */

/* Collaborating Class Declarations */
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/DTRecHit/interface/DTSLRecCluster.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"

/* C++ Headers */
#include <functional>

/* ====================================================================== */

/* Class DTRecClusterCollection Interface */
typedef edm::RangeMap<DTSuperLayerId, edm::OwnVector<DTSLRecCluster> > DTRecClusterCollection;

#endif // DTRECHIT_DTRECCLUSTERCOLLECTION_H

