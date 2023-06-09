/** \class MuRecoFlatTableProducers.ccMuRecoFlatTableProducers DPGAnalysis/MuonTools/src/MuRecoFlatTableProducers.cc
 *  
 * EDProducers : the flat table producers for DT, GEM and RPC RecHits and Segments
 *
 * \author C. Battilana (INFN BO)
 *
 *
 */

#include "DPGAnalysis/MuonTools/interface/MuLocalRecoBaseProducer.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

using DTSegmentFlatTableProducer = MuRecObjBaseProducer<DTChamberId, DTRecSegment4D, DTGeometry>;

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

using RPCRecHitFlatTableProducer = MuRecObjBaseProducer<RPCDetId, RPCRecHit, RPCGeometry>;

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"

using GEMRecHitFlatTableProducer = MuRecObjBaseProducer<GEMDetId, GEMRecHit, GEMGeometry>;

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/GEMRecHit/interface/GEMSegmentCollection.h"

using GEMSegmentFlatTableProducer = MuRecObjBaseProducer<GEMDetId, GEMSegment, GEMGeometry>;

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(DTSegmentFlatTableProducer);
DEFINE_FWK_MODULE(RPCRecHitFlatTableProducer);
DEFINE_FWK_MODULE(GEMRecHitFlatTableProducer);
DEFINE_FWK_MODULE(GEMSegmentFlatTableProducer);
