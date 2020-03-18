// Our own includes
#include "RecoLocalFastTime/FTLClusterizer/interface/MTDArrayBuffer.h"
#include "RecoLocalFastTime/FTLClusterizer/interface/MTDThresholdClusterizer.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

// STL
#include <stack>
#include <vector>
#include <iostream>
#include <atomic>
#include <cmath>
using namespace std;

//#define DEBUG_ENABLED
#ifdef DEBUG_ENABLED
#define DEBUG(x)                 \
  do {                           \
    std::cout << x << std::endl; \
  } while (0)
#else
#define DEBUG(x)
#endif

//----------------------------------------------------------------------------
//! Constructor:
//----------------------------------------------------------------------------
MTDThresholdClusterizer::MTDThresholdClusterizer(edm::ParameterSet const& conf)
    :  // Get energy thresholds
      theHitThreshold(conf.getParameter<double>("HitThreshold")),
      theSeedThreshold(conf.getParameter<double>("SeedThreshold")),
      theClusterThreshold(conf.getParameter<double>("ClusterThreshold")),
      theTimeThreshold(conf.getParameter<double>("TimeThreshold")),
      thePositionThreshold(conf.getParameter<double>("PositionThreshold")),
      BTLBarLength(conf.getParameter<double>("BTLBarLength")),
      theNumOfRows(0),
      theNumOfCols(0),
      theCurrentId(0),
      theBuffer(theNumOfRows, theNumOfCols),
      bufferAlreadySet(false) {}
/////////////////////////////////////////////////////////////////////////////
MTDThresholdClusterizer::~MTDThresholdClusterizer() {}

// Configuration descriptions
void MTDThresholdClusterizer::fillDescriptions(edm::ParameterSetDescription& desc) {
  desc.add<double>("HitThreshold", 0.);
  desc.add<double>("SeedThreshold", 0.);
  desc.add<double>("ClusterThreshold", 0.);
  desc.add<double>("TimeThreshold", 10.);
  desc.add<double>("PositionThreshold", 4.);
  desc.add<double>("BTLBarLength", 57.0);  //length of bar in mm
}

//----------------------------------------------------------------------------
//!  Prepare the Clusterizer to work on a particular DetUnit.  Re-init the
//!  size of the panel/plaquette (so update nrows and ncols),
//----------------------------------------------------------------------------
bool MTDThresholdClusterizer::setup(const MTDGeometry* geom, const MTDTopology* topo, const DetId& id) {
  theCurrentId = id;
  //using geopraphicalId here
  const auto& thedet = geom->idToDet(id);
  if (thedet == nullptr) {
    throw cms::Exception("MTDThresholdClusterizer")
        << "GeographicalID: " << std::hex << id.rawId() << " is invalid!" << std::dec << std::endl;
    return false;
  }
  const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
  const RectangularMTDTopology& topol = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

  // Get the new sizes.
  unsigned int nrows = topol.nrows();     // rows in x
  unsigned int ncols = topol.ncolumns();  // cols in y

  theNumOfRows = nrows;  // Set new sizes
  theNumOfCols = ncols;

  DEBUG("Buffer size [" << theNumOfRows + 1 << "," << theNumOfCols + 1 << "]");

  if (nrows > theBuffer.rows() || ncols > theBuffer.columns()) {  // change only when a larger is needed
    // Resize the buffer
    theBuffer.setSize(nrows + 1, ncols + 1);  // +1 needed for MTD
    bufferAlreadySet = true;
  }

  return true;
}
//----------------------------------------------------------------------------
//!  \brief Cluster hits.
//!  This method operates on a matrix of hits
//!  and finds the largest contiguous cluster around
//!  each seed hit.
//!  Input and output data stored in DetSet
//----------------------------------------------------------------------------
void MTDThresholdClusterizer::clusterize(const FTLRecHitCollection& input,
                                         const MTDGeometry* geom,
                                         const MTDTopology* topo,
                                         FTLClusterCollection& output) {
  FTLRecHitCollection::const_iterator begin = input.begin();
  FTLRecHitCollection::const_iterator end = input.end();

  // Do not bother for empty detectors
  if (begin == end) {
    edm::LogInfo("MTDThresholdClusterizer") << "No hits to clusterize";
    return;
  }

  DEBUG("Input collection " << input.size());
  assert(output.empty());

  std::set<unsigned> geoIds;
  std::multimap<unsigned, unsigned> geoIdToIdx;

  unsigned index = 0;
  for (const auto& hit : input) {
    MTDDetId mtdId = MTDDetId(hit.detid());
    if (mtdId.subDetector() != MTDDetId::FastTime) {
      throw cms::Exception("MTDThresholdClusterizer")
          << "MTDDetId: " << std::hex << mtdId.rawId() << " is invalid!" << std::dec << std::endl;
    }

    if (mtdId.mtdSubDetector() == MTDDetId::BTL) {
      BTLDetId hitId(hit.detid());
      DetId geoId = hitId.geographicalId(
          (BTLDetId::CrysLayout)topo->getMTDTopologyMode());  //for BTL topology gives different layout id
      geoIdToIdx.emplace(geoId, index);
      geoIds.emplace(geoId);
      ++index;
    } else if (mtdId.mtdSubDetector() == MTDDetId::ETL) {
      ETLDetId hitId(hit.detid());
      DetId geoId = hitId.geographicalId();
      geoIdToIdx.emplace(geoId, index);
      geoIds.emplace(geoId);
      ++index;
    } else {
      throw cms::Exception("MTDThresholdClusterizer")
          << "MTDDetId: " << std::hex << mtdId.rawId() << " is invalid!" << std::dec << std::endl;
    }
  }

  //cluster hits within geoIds (modules)
  for (unsigned id : geoIds) {
    //  Set up the clusterization on this DetId.
    if (!setup(geom, topo, DetId(id)))
      return;

    auto range = geoIdToIdx.equal_range(id);
    DEBUG("Matching Ids for " << std::hex << id << std::dec << " [" << range.first->second << ","
                              << range.second->second << "]");

    //  Copy MTDRecHits to the buffer array; select the seed hits
    //  on the way, and store them in theSeeds.
    for (auto itr = range.first; itr != range.second; ++itr) {
      const unsigned hitidx = itr->second;
      copy_to_buffer(begin + hitidx, geom, topo);
    }

    FTLClusterCollection::FastFiller clustersOnDet(output, id);

    for (unsigned int i = 0; i < theSeeds.size(); i++) {
      if (theBuffer.energy(theSeeds[i]) > theSeedThreshold) {  // Is this seed still valid?
        //  Make a cluster around this seed
        const FTLCluster& cluster = make_cluster(theSeeds[i]);

        //  Check if the cluster is above threshold
        if (cluster.energy() > theClusterThreshold) {
          DEBUG("putting in this cluster " << i << " #hits:" << cluster.size() << " E:" << cluster.energy()
                                           << " T:" << cluster.time() << " X:" << cluster.x() << " Y:" << cluster.y());
          clustersOnDet.push_back(cluster);
        }
      }
    }

    // Erase the seeds.
    theSeeds.clear();
    //  Need to clean unused hits from the buffer array.
    for (auto itr = range.first; itr != range.second; ++itr) {
      const unsigned hitidx = itr->second;
      clear_buffer(begin + hitidx);
    }
  }
}

//----------------------------------------------------------------------------
//!  \brief Clear the internal buffer array.
//!
//!  MTDs which are not part of recognized clusters are NOT ERASED
//!  during the cluster finding.  Erase them now.
//!
//----------------------------------------------------------------------------
void MTDThresholdClusterizer::clear_buffer(RecHitIterator itr) { theBuffer.clear(itr->row(), itr->column()); }

//----------------------------------------------------------------------------
//! \brief Copy FTLRecHit into the buffer, identify seeds.
//----------------------------------------------------------------------------
void MTDThresholdClusterizer::copy_to_buffer(RecHitIterator itr, const MTDGeometry* geom, const MTDTopology* topo) {
  MTDDetId mtdId = MTDDetId(itr->detid());
  int row = itr->row();
  int col = itr->column();
  GeomDetEnumerators::Location SubDet = GeomDetEnumerators::invalidLoc;
  float energy = itr->energy();
  float time = itr->time();
  float timeError = itr->timeError();
  float position = itr->position().second;
  LocalError local_error(0, 0, 0);
  GlobalPoint global_point(0, 0, 0);
  if (mtdId.mtdSubDetector() == MTDDetId::BTL) {
    SubDet = GeomDetEnumerators::barrel;
    BTLDetId id = itr->id();
    DetId geoId = id.geographicalId((BTLDetId::CrysLayout)topo->getMTDTopologyMode());
    const auto& det = geom->idToDet(geoId);
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(det->topology());
    const RectangularMTDTopology& topol = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());
    MeasurementPoint mp(row, col);
    LocalPoint lp_ctr = topol.localPosition(mp);
    float z_offset = ((BTLBarLength / 2.) - position) * 1._mm;
    //need to add the offset as the distance from centre to edge (z direction in global and y direction in local)
    LocalPoint lp(lp_ctr.x(), lp_ctr.y() + z_offset, lp_ctr.z());
    global_point = det->toGlobal(lp);
    BTLRecHitsErrorEstimatorIM I(det, lp);
    LocalError err_modified = I.localError();
    local_error = err_modified;
  } else if (mtdId.mtdSubDetector() == MTDDetId::ETL) {
    SubDet = GeomDetEnumerators::endcap;
    ETLDetId id = itr->id();
    DetId geoId = id.geographicalId();
    const auto& det = geom->idToDet(geoId);
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(det->topology());
    const RectangularMTDTopology& topol = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

    MeasurementPoint mp(row, col);
    LocalPoint lp = topol.localPosition(mp);
    global_point = det->toGlobal(lp);
  }

  DEBUG("ROW " << row << " COL " << col << " ENERGY " << energy << " TIME " << time);
  if (energy > theHitThreshold) {
    theBuffer.set(row, col, SubDet, energy, time, timeError, local_error, global_point);
    if (energy > theSeedThreshold)
      theSeeds.push_back(FTLCluster::FTLHitPos(row, col));
    //sort seeds?
  }
}

//----------------------------------------------------------------------------
//!  \brief The actual clustering algorithm: group the neighboring hits around the seed.
//----------------------------------------------------------------------------
FTLCluster MTDThresholdClusterizer::make_cluster(const FTLCluster::FTLHitPos& hit) {
  //First we acquire the seeds for the clusters

  GeomDetEnumerators::Location seed_subdet = theBuffer.SubDet(hit.row(), hit.col());
  float seed_energy = theBuffer.energy(hit.row(), hit.col());
  float seed_time = theBuffer.time(hit.row(), hit.col());
  float seed_time_error = theBuffer.time_error(hit.row(), hit.col());
  double seed_z = theBuffer.global_point(hit.row(), hit.col()).z();
  double seed_x = theBuffer.global_point(hit.row(), hit.col()).x();
  double seed_y = theBuffer.global_point(hit.row(), hit.col()).y();
  double seed_error_xx = theBuffer.local_error(hit.row(), hit.col()).xx();
  double seed_error_yy = theBuffer.local_error(hit.row(), hit.col()).yy();
  theBuffer.clear(hit);

  AccretionCluster acluster;
  acluster.add(hit, seed_energy, seed_time, seed_time_error);

  bool stopClus = false;
  //Here we search all hits adjacent to all hits in the cluster.
  while (!acluster.empty() && !stopClus) {
    //This is the standard algorithm to find and add a hit
    auto curInd = acluster.top();
    acluster.pop();
    for (auto c = std::max(0, int(acluster.y[curInd] - 1));
         c < std::min(int(acluster.y[curInd] + 2), int(theBuffer.columns())) && !stopClus;
         ++c) {
      for (auto r = std::max(0, int(acluster.x[curInd] - 1));
           r < std::min(int(acluster.x[curInd] + 2), int(theBuffer.rows())) && !stopClus;
           ++r) {
        if (theBuffer.energy(r, c) > theHitThreshold) {
          if (std::abs(theBuffer.time(r, c) - seed_time) >
              theTimeThreshold *
                  sqrt(theBuffer.time_error(r, c) * theBuffer.time_error(r, c) + seed_time_error * seed_time_error))
            continue;
          if ((seed_subdet == GeomDetEnumerators::barrel) && (theBuffer.SubDet(r, c) == GeomDetEnumerators::barrel)) {
            double x_hit = theBuffer.global_point(r, c).x();
            double y_hit = theBuffer.global_point(r, c).y();
            double z_hit = theBuffer.global_point(r, c).z();
            double hit_error_xx = theBuffer.local_error(r, c).xx();
            double hit_error_yy = theBuffer.local_error(r, c).yy();
            if (thePositionThreshold > 0) {
              if (sqrt(pow((x_hit - seed_x), 2) + pow((y_hit - seed_y), 2) + pow((z_hit - seed_z), 2)) >
                  thePositionThreshold * sqrt(hit_error_xx + seed_error_xx + hit_error_yy + seed_error_yy))
                continue;
            }
          }
          FTLCluster::FTLHitPos newhit(r, c);
          if (!acluster.add(newhit, theBuffer.energy(r, c), theBuffer.time(r, c), theBuffer.time_error(r, c))) {
            stopClus = true;
            break;
          }
          theBuffer.clear(newhit);
        }
      }
    }
  }  // while accretion

  FTLCluster cluster(theCurrentId,
                     acluster.isize,
                     acluster.energy.data(),
                     acluster.time.data(),
                     acluster.timeError.data(),
                     acluster.x.data(),
                     acluster.y.data(),
                     acluster.xmin,
                     acluster.ymin);
  return cluster;
}
