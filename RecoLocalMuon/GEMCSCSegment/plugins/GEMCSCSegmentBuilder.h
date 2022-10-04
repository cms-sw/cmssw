#ifndef GEMCSCSegment_GEMCSCSegmentBuilder_h
#define GEMCSCSegment_GEMCSCSegmentBuilder_h

/** \class GEMCSCSegmentBuilder 
 *
 * Algorithm to build GEMCSCSegments from GEMRecHit and CSCSegment collections
 * by implementing a 'build' function required by GEMCSCSegmentProducer.
 *
 *
 * $Date:  $
 * $Revision: 1.3 $
 * \author Raffaella Radogna
 *
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/GEMGeometry/interface/GEMGeometry.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <Geometry/GEMGeometry/interface/GEMEtaPartition.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include <DataFormats/GEMRecHit/interface/GEMRecHit.h>
#include <DataFormats/GEMRecHit/interface/GEMRecHitCollection.h>
#include <DataFormats/GEMRecHit/interface/GEMCSCSegmentCollection.h>

class CSCStationIndex {
public:
  CSCStationIndex() : _region(0), _station(0), _ring(0), _chamber(0), _layer(0){};
  CSCStationIndex(int region, int station, int ring, int chamber, int layer)
      : _region(region), _station(station), _ring(ring), _chamber(chamber), _layer(layer) {}
  ~CSCStationIndex() {}

  int region() const { return _region; }
  int station() const { return _station; }
  int ring() const { return _ring; }
  int chamber() const { return _chamber; }
  int layer() const { return _layer; }

  bool operator<(const CSCStationIndex& cscind) const {
    if (cscind.region() != this->region())
      return cscind.region() < this->region();
    else if (cscind.station() != this->station())
      return cscind.station() < this->station();
    else if (cscind.ring() != this->ring())
      return cscind.ring() < this->ring();
    else if (cscind.chamber() != this->chamber())
      return cscind.chamber() < this->chamber();
    else if (cscind.layer() != this->layer())
      return cscind.layer() < this->layer();
    return false;
  }

private:
  int _region;
  int _station;
  int _ring;
  int _chamber;
  int _layer;
};

class GEMCSCSegmentAlgorithm;

class GEMCSCSegmentBuilder {
public:
  explicit GEMCSCSegmentBuilder(const edm::ParameterSet&);
  ~GEMCSCSegmentBuilder();
  void build(const GEMRecHitCollection* rechits, const CSCSegmentCollection* cscsegments, GEMCSCSegmentCollection& oc);

  void setGeometry(const GEMGeometry* gemgeom, const CSCGeometry* cscgeom);
  void LinkGEMRollsToCSCChamberIndex(const GEMGeometry* gemgeom, const CSCGeometry* cscgeom);

protected:
  std::map<CSCStationIndex, std::set<GEMDetId> > rollstoreCSC;

private:
  std::unique_ptr<GEMCSCSegmentAlgorithm> algo;
  const bool enable_me21_ge21_;
  const GEMGeometry* gemgeom_;
  const CSCGeometry* cscgeom_;
};

#endif
