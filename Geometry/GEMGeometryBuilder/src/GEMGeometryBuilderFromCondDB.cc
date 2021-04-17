/** Implementation of the GEM Geometry Builder from GEM record stored in CondDB
 *
 *  \author M. Maggi - INFN Bari
 */
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/GEMGeometryBuilder/src/GEMGeometryBuilderFromCondDB.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"

#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "DataFormats/Math/interface/GeantUnits.h"

#include <algorithm>

using namespace geant_units::operators;

GEMGeometryBuilderFromCondDB::GEMGeometryBuilderFromCondDB() {}

GEMGeometryBuilderFromCondDB::~GEMGeometryBuilderFromCondDB() {}

void GEMGeometryBuilderFromCondDB::build(GEMGeometry& theGeometry, const RecoIdealGeometry& rgeo) {
  const std::vector<DetId>& detids(rgeo.detIds());
  std::unordered_map<uint32_t, GEMSuperChamber*> superChambers;
  std::unordered_map<uint32_t, GEMChamber*> chambers;
  std::unordered_map<uint32_t, GEMEtaPartition*> partitions;

  for (unsigned int id = 0; id < detids.size(); ++id) {
    GEMDetId gemid(detids[id]);
    LogDebug("GEMGeometryBuilder") << "GEMGeometryBuilder adding " << gemid;
    if (gemid.roll() == 0) {
      if (gemid.layer() == 0) {
        GEMSuperChamber* gsc = buildSuperChamber(rgeo, id, gemid);
        superChambers.emplace(gemid.rawId(), gsc);
      } else {
        GEMChamber* gch = buildChamber(rgeo, id, gemid);
        chambers.emplace(gemid.rawId(), gch);
      }
    } else {
      GEMEtaPartition* gep = buildEtaPartition(rgeo, id, gemid);
      partitions.emplace(gemid.rawId(), gep);
    }
  }

  ////////////////////////////////////////////////////////////
  // TEMP - for backward compatability with old geometry
  // no superchambers or chambers in old geometry, using etpartitions
  if (superChambers.empty()) {
    for (unsigned int id = 0; id < detids.size(); ++id) {
      GEMDetId gemid(detids[id]);
      if (gemid.roll() == 1) {
        GEMChamber* gch = buildChamber(rgeo, id, gemid.chamberId());
        chambers.emplace(gemid.chamberId().rawId(), gch);
        if (gemid.layer() == 1) {
          GEMSuperChamber* gsc = buildSuperChamber(rgeo, id, gemid.superChamberId());
          superChambers.emplace(gemid.superChamberId().rawId(), gsc);
        }
      }
    }
  }
  ////////////////////////////////////////////////////////////

  // construct the regions, stations and rings.
  for (int re = -1; re <= 1; re = re + 2) {
    GEMRegion* region = new GEMRegion(re);

    for (int st = 0; st <= GEMDetId::maxStationId; ++st) {
      GEMStation* station = new GEMStation(re, st);
      std::string sign(re == -1 ? "-" : "");
      std::string name("GE" + sign + std::to_string(st) + "/1");
      station->setName(name);

      for (int ri = 1; ri <= 1; ++ri) {
        GEMRing* ring = new GEMRing(re, st, ri);

        for (auto sch : superChambers) {
          auto superChamber = sch.second;
          const GEMDetId scId(superChamber->id());
          if (scId.region() != re || scId.station() != st || scId.ring() != ri)
            continue;
          int ch = scId.chamber();

          for (int ly = 1; ly <= GEMDetId::maxLayerId; ++ly) {
            const GEMDetId chId(re, ri, st, ly, ch, 0);
            auto chamberIt = chambers.find(chId.rawId());
            if (chamberIt == chambers.end())
              continue;
            auto chamber = chamberIt->second;

            for (int roll = 1; roll <= GEMDetId::maxRollId; ++roll) {
              const GEMDetId rollId(re, ri, st, ly, ch, roll);
              auto gepIt = partitions.find(rollId.rawId());
              if (gepIt == partitions.end())
                continue;
              auto gep = gepIt->second;

              chamber->add(gep);
              theGeometry.add(gep);
            }

            superChamber->add(chamber);
            theGeometry.add(chamber);
          }

          LogDebug("GEMGeometryBuilder") << "Adding super chamber " << scId << " to ring:";
          ring->add(superChamber);
          theGeometry.add(superChamber);
        }  // end superChambers

        if (ring->nSuperChambers()) {
          LogDebug("GEMGeometryBuilder") << "Adding ring " << ri << " to station "
                                         << "re " << re << " st " << st;
          station->add(ring);
          theGeometry.add(ring);
        } else {
          delete ring;
        }
      }  // end ring

      if (station->nRings()) {
        LogDebug("GEMGeometryBuilder") << "Adding station " << st << " to region " << re;
        region->add(station);
        theGeometry.add(station);
      } else {
        delete station;
      }
    }  // end station

    LogDebug("GEMGeometryBuilder") << "Adding region " << re << " to the geometry";
    theGeometry.add(region);
  }
}

GEMSuperChamber* GEMGeometryBuilderFromCondDB::buildSuperChamber(const RecoIdealGeometry& rgeo,
                                                                 unsigned int gid,
                                                                 GEMDetId detId) const {
  LogDebug("GEMGeometryBuilderFromCondDB") << "buildSuperChamber " << detId;

  RCPBoundPlane surf(boundPlane(rgeo, gid, detId));

  GEMSuperChamber* superChamber = new GEMSuperChamber(detId, surf);
  return superChamber;
}

GEMChamber* GEMGeometryBuilderFromCondDB::buildChamber(const RecoIdealGeometry& rgeo,
                                                       unsigned int gid,
                                                       GEMDetId detId) const {
  LogDebug("GEMGeometryBuilderFromCondDB") << "buildChamber " << detId;

  RCPBoundPlane surf(boundPlane(rgeo, gid, detId));

  GEMChamber* chamber = new GEMChamber(detId, surf);
  return chamber;
}

GEMEtaPartition* GEMGeometryBuilderFromCondDB::buildEtaPartition(const RecoIdealGeometry& rgeo,
                                                                 unsigned int gid,
                                                                 GEMDetId detId) const {
  std::vector<std::string>::const_iterator strStart = rgeo.strStart(gid);
  std::string name = *(strStart);
  LogDebug("GEMGeometryBuilderFromCondDB") << "buildEtaPartition " << name << " " << detId;

  std::vector<double>::const_iterator shapeStart = rgeo.shapeStart(gid);
  float be = convertMmToCm(*(shapeStart + 0));
  float te = convertMmToCm(*(shapeStart + 1));
  float ap = convertMmToCm(*(shapeStart + 2));
  float ti = convertMmToCm(*(shapeStart + 3));
  float nstrip = *(shapeStart + 4);
  float npad = *(shapeStart + 5);
  //float dphi = *(shapeStart + 6);

  std::vector<float> pars;
  pars.emplace_back(be);
  pars.emplace_back(te);
  pars.emplace_back(ap);
  pars.emplace_back(nstrip);
  pars.emplace_back(npad);
  //pars.emplace_back(dphi);
  pars.emplace_back(0.17715);  //temporary input for PR

  RCPBoundPlane surf(boundPlane(rgeo, gid, detId));
  GEMEtaPartitionSpecs* e_p_specs = new GEMEtaPartitionSpecs(GeomDetEnumerators::GEM, name, pars);

  LogDebug("GEMGeometryBuilderFromCondDB") << "size " << be << " " << te << " " << ap << " " << ti;
  GEMEtaPartition* etaPartition = new GEMEtaPartition(detId, surf, e_p_specs);
  return etaPartition;
}

GEMGeometryBuilderFromCondDB::RCPBoundPlane GEMGeometryBuilderFromCondDB::boundPlane(const RecoIdealGeometry& rgeo,
                                                                                     unsigned int gid,
                                                                                     GEMDetId detId) const {
  std::vector<double>::const_iterator shapeStart = rgeo.shapeStart(gid);
  float be = convertMmToCm(*(shapeStart + 0));
  float te = convertMmToCm(*(shapeStart + 1));
  float ap = convertMmToCm(*(shapeStart + 2));
  float ti = convertMmToCm(*(shapeStart + 3));
  Bounds* bounds = new TrapezoidalPlaneBounds(be, te, ap, ti);

  std::vector<double>::const_iterator tranStart = rgeo.tranStart(gid);
  Surface::PositionType posResult(
      convertMmToCm(*(tranStart)), convertMmToCm(*(tranStart + 1)), convertMmToCm(*(tranStart + 2)));

  std::vector<double>::const_iterator rotStart = rgeo.rotStart(gid);
  Surface::RotationType rotResult(*(rotStart + 0),
                                  *(rotStart + 1),
                                  *(rotStart + 2),
                                  *(rotStart + 3),
                                  *(rotStart + 4),
                                  *(rotStart + 5),
                                  *(rotStart + 6),
                                  *(rotStart + 7),
                                  *(rotStart + 8));

  //Change of axes for the forward
  Basic3DVector<float> newX(1., 0., 0.);
  Basic3DVector<float> newY(0., 0., -1.);
  Basic3DVector<float> newZ(0., 1., 0.);

  rotResult.rotateAxes(newX, newY, newZ);

  return RCPBoundPlane(new BoundPlane(posResult, rotResult, bounds));
}
