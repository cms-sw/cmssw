#include "L1Trigger/L1TMuon/interface/GeometryTranslator.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "L1Trigger/DTUtilities/interface/DTTrigGeom.h"
#include "DataFormats/CSCDigi/interface/CSCConstants.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCPatternBank.h"

#include "L1Trigger/L1TMuon/interface/MuonTriggerPrimitive.h"

#include <cmath>  // for pi

using namespace L1TMuon;

GeometryTranslator::GeometryTranslator(edm::ConsumesCollector iC)
    : _geom_cache_id(0ULL),
      geodtToken_(iC.esConsumes()),
      geocscToken_(iC.esConsumes()),
      georpcToken_(iC.esConsumes()),
      geogemToken_(iC.esConsumes()),
      geome0Token_(iC.esConsumes()),
      _magfield_cache_id(0ULL),
      magfieldToken_(iC.esConsumes()) {}

GeometryTranslator::~GeometryTranslator() {}

double GeometryTranslator::calculateGlobalEta(const TriggerPrimitive& tp) const {
  switch (tp.subsystem()) {
    case L1TMuon::kDT:
      return calcDTSpecificEta(tp);
      break;
    case L1TMuon::kCSC:
      return calcCSCSpecificEta(tp);
      break;
    case L1TMuon::kRPC:
      return calcRPCSpecificEta(tp);
      break;
    case L1TMuon::kGEM:
      return calcGEMSpecificEta(tp);
      break;
    case L1TMuon::kME0:
      return calcME0SpecificEta(tp);
      break;
    default:
      return std::nan("Invalid TP type!");
      break;
  }
}

double GeometryTranslator::calculateGlobalPhi(const TriggerPrimitive& tp) const {
  switch (tp.subsystem()) {
    case L1TMuon::kDT:
      return calcDTSpecificPhi(tp);
      break;
    case L1TMuon::kCSC:
      return calcCSCSpecificPhi(tp);
      break;
    case L1TMuon::kRPC:
      return calcRPCSpecificPhi(tp);
      break;
    case L1TMuon::kGEM:
      return calcGEMSpecificPhi(tp);
      break;
    case L1TMuon::kME0:
      return calcME0SpecificPhi(tp);
      break;
    default:
      return std::nan("Invalid TP type!");
      break;
  }
}

double GeometryTranslator::calculateBendAngle(const TriggerPrimitive& tp) const {
  switch (tp.subsystem()) {
    case L1TMuon::kDT:
      return calcDTSpecificBend(tp);
      break;
    case L1TMuon::kCSC:
      return calcCSCSpecificBend(tp);
      break;
    case L1TMuon::kRPC:
      return calcRPCSpecificBend(tp);
      break;
    case L1TMuon::kGEM:
      return calcGEMSpecificBend(tp);
      break;
    case L1TMuon::kME0:
      return calcME0SpecificBend(tp);
      break;
    default:
      return std::nan("Invalid TP type!");
      break;
  }
}

GlobalPoint GeometryTranslator::getGlobalPoint(const TriggerPrimitive& tp) const {
  switch (tp.subsystem()) {
    case L1TMuon::kDT:
      return calcDTSpecificPoint(tp);
      break;
    case L1TMuon::kCSC:
      return getCSCSpecificPoint(tp);
      break;
    case L1TMuon::kRPC:
      return getRPCSpecificPoint(tp);
      break;
    case L1TMuon::kGEM:
      return getGEMSpecificPoint(tp);
      break;
    case L1TMuon::kME0:
      return getME0SpecificPoint(tp);
      break;
    default:
      GlobalPoint ret(
          GlobalPoint::Polar(std::nan("Invalid TP type!"), std::nan("Invalid TP type!"), std::nan("Invalid TP type!")));
      return ret;
      break;
  }
}

void GeometryTranslator::checkAndUpdateGeometry(const edm::EventSetup& es) {
  const MuonGeometryRecord& geom = es.get<MuonGeometryRecord>();
  unsigned long long geomid = geom.cacheIdentifier();
  if (_geom_cache_id != geomid) {
    _geodt = geom.getHandle(geodtToken_);
    _geocsc = geom.getHandle(geocscToken_);
    _georpc = geom.getHandle(georpcToken_);
    _geogem = geom.getHandle(geogemToken_);
    _geome0 = geom.getHandle(geome0Token_);
    _geom_cache_id = geomid;
  }

  const IdealMagneticFieldRecord& magfield = es.get<IdealMagneticFieldRecord>();
  unsigned long long magfieldid = magfield.cacheIdentifier();
  if (_magfield_cache_id != magfieldid) {
    _magfield = magfield.getHandle(magfieldToken_);
    _magfield_cache_id = magfieldid;
  }
}

// _____________________________________________________________________________
// ME0
GlobalPoint GeometryTranslator::getME0SpecificPoint(const TriggerPrimitive& tp) const {
  if (tp.detId<DetId>().subdetId() == MuonSubdetId::GEM) { // GE0
    const GEMDetId id(tp.detId<GEMDetId>());
    const GEMSuperChamber* chamber = _geogem->superChamber(id);
    const GEMChamber* keylayer = chamber->chamber(3);  // GEM key layer is layer 3
    int partition = tp.getME0Data().partition;     // 'partition' is in half-roll unit
    int iroll = (partition >> 1) + 1;
    const GEMEtaPartition* roll = keylayer->etaPartition(iroll);
    assert(roll != nullptr);  // failed to get GEM roll
    // See L1Trigger/ME0Trigger/src/ME0TriggerPseudoBuilder.cc
    int phiposition = tp.getME0Data().phiposition;  // 'phiposition' is in half-strip unit
    int istrip = (phiposition >> 1);
    int phiposition2 = (phiposition & 0x1);
    float centreOfStrip = istrip + 0.25 + phiposition2 * 0.5;
    const LocalPoint& lp = roll->centreOfStrip(centreOfStrip);
    const GlobalPoint& gp = roll->surface().toGlobal(lp);
    return gp;
  } else { // ME0
    const ME0DetId id(tp.detId<ME0DetId>());
    const ME0Chamber* chamber = _geome0->chamber(id);
    const ME0Layer* keylayer = chamber->layer(3);  // ME0 key layer is layer 3
    int partition = tp.getME0Data().partition;     // 'partition' is in half-roll unit
    int iroll = (partition >> 1) + 1;
    const ME0EtaPartition* roll = keylayer->etaPartition(iroll);
    assert(roll != nullptr);  // failed to get ME0 roll
    // See L1Trigger/ME0Trigger/src/ME0TriggerPseudoBuilder.cc
    int phiposition = tp.getME0Data().phiposition;  // 'phiposition' is in half-strip unit
    int istrip = (phiposition >> 1);
    int phiposition2 = (phiposition & 0x1);
    float centreOfStrip = istrip + 0.25 + phiposition2 * 0.5;
    const LocalPoint& lp = roll->centreOfStrip(centreOfStrip);
    const GlobalPoint& gp = roll->surface().toGlobal(lp);
    return gp;
  }
}

double GeometryTranslator::calcME0SpecificEta(const TriggerPrimitive& tp) const {
  return getME0SpecificPoint(tp).eta();
}

double GeometryTranslator::calcME0SpecificPhi(const TriggerPrimitive& tp) const {
  return getME0SpecificPoint(tp).phi();
}

double GeometryTranslator::calcME0SpecificBend(const TriggerPrimitive& tp) const {
  return tp.getME0Data().deltaphi * (tp.getME0Data().bend == 0 ? 1 : -1);
}

// _____________________________________________________________________________
// GEM
GlobalPoint GeometryTranslator::getGEMSpecificPoint(const TriggerPrimitive& tp) const {
  const GEMDetId id(tp.detId<GEMDetId>());
  const GEMEtaPartition* roll = _geogem->etaPartition(id);
  assert(roll != nullptr);  // failed to get GEM roll
  //const uint16_t pad = tp.getGEMData().pad;
  // Use half-pad precision, + 0.5 at the end to get the center of the pad (pad starts from 0)
  const float pad = (0.5 * static_cast<float>(tp.getGEMData().pad_low + tp.getGEMData().pad_hi)) + 0.5f;
  const LocalPoint& lp = roll->centreOfPad(pad);
  const GlobalPoint& gp = roll->surface().toGlobal(lp);
  return gp;
}

double GeometryTranslator::calcGEMSpecificEta(const TriggerPrimitive& tp) const {
  return getGEMSpecificPoint(tp).eta();
}

double GeometryTranslator::calcGEMSpecificPhi(const TriggerPrimitive& tp) const {
  return getGEMSpecificPoint(tp).phi();
}

double GeometryTranslator::calcGEMSpecificBend(const TriggerPrimitive& tp) const { return 0.0; }

// _____________________________________________________________________________
// RPC
GlobalPoint GeometryTranslator::getRPCSpecificPoint(const TriggerPrimitive& tp) const {
  const RPCDetId id(tp.detId<RPCDetId>());
  const RPCRoll* roll = _georpc->roll(id);
  assert(roll != nullptr);  // failed to get RPC roll
  //const int strip = static_cast<int>(tp.getRPCData().strip);
  // Use half-strip precision, - 0.5 at the end to get the center of the strip (strip starts from 1)
  const float strip = (0.5 * static_cast<float>(tp.getRPCData().strip_low + tp.getRPCData().strip_hi)) - 0.5f;
  const LocalPoint& lp = roll->centreOfStrip(strip);
  const GlobalPoint& gp = roll->surface().toGlobal(lp);
  return gp;
}

double GeometryTranslator::calcRPCSpecificEta(const TriggerPrimitive& tp) const {
  return getRPCSpecificPoint(tp).eta();
}

double GeometryTranslator::calcRPCSpecificPhi(const TriggerPrimitive& tp) const {
  return getRPCSpecificPoint(tp).phi();
}

// this function actually does nothing since RPC
// hits are point-like objects
double GeometryTranslator::calcRPCSpecificBend(const TriggerPrimitive& tp) const { return 0.0; }

// _____________________________________________________________________________
// CSC
//
// alot of this is transcription and consolidation of the CSC
// global phi calculation code
// this works directly with the geometry
// rather than using the old phi luts
GlobalPoint GeometryTranslator::getCSCSpecificPoint(const TriggerPrimitive& tp) const {
  const CSCDetId id(tp.detId<CSCDetId>());
  // we should change this to weak_ptrs at some point
  // requires introducing std::shared_ptrs to geometry
  std::unique_ptr<const CSCChamber> chamb(_geocsc->chamber(id));
  assert(chamb != nullptr);  // failed to get CSC chamber
  std::unique_ptr<const CSCLayerGeometry> layer_geom(chamb->layer(CSCConstants::KEY_ALCT_LAYER)->geometry());
  std::unique_ptr<const CSCLayer> layer(chamb->layer(CSCConstants::KEY_ALCT_LAYER));

  const uint16_t halfstrip = tp.getCSCData().strip;
  const uint16_t pattern = tp.getCSCData().pattern;
  const uint16_t keyWG = tp.getCSCData().keywire;
  //const unsigned maxStrips = layer_geom->numberOfStrips();

  // so we can extend this later
  // assume TMB2007 half-strips only as baseline
  double offset = 0.0;
  switch (1) {
    case 1:
      offset = CSCPatternBank::getLegacyPosition(pattern);
  }
  const unsigned halfstrip_offs = static_cast<unsigned>(0.5 + halfstrip + offset);
  const unsigned strip = halfstrip_offs / 2 + 1;  // geom starts from 1

  // the rough location of the hit at the ALCT key layer
  // we will refine this using the half strip information
  const LocalPoint& coarse_lp = layer_geom->stripWireGroupIntersection(strip, keyWG);
  const GlobalPoint& coarse_gp = layer->surface().toGlobal(coarse_lp);

  // the strip width/4.0 gives the offset of the half-strip
  // center with respect to the strip center
  const double hs_offset = layer_geom->stripPhiPitch() / 4.0;

  // determine handedness of the chamber
  const bool ccw = isCSCCounterClockwise(layer);
  // we need to subtract the offset of even half strips and add the odd ones
  const double phi_offset = ((halfstrip_offs % 2 ? 1 : -1) * (ccw ? -hs_offset : hs_offset));

  // the global eta calculation uses the middle of the strip
  // so no need to increment it
  const GlobalPoint final_gp(
      GlobalPoint::Polar(coarse_gp.theta(), (coarse_gp.phi().value() + phi_offset), coarse_gp.mag()));

  // We need to add in some notion of the 'error' on trigger primitives
  // like the width of the wire group by the width of the strip
  // or something similar

  // release ownership of the pointers
  chamb.release();
  layer_geom.release();
  layer.release();

  return final_gp;
}

double GeometryTranslator::calcCSCSpecificEta(const TriggerPrimitive& tp) const {
  return getCSCSpecificPoint(tp).eta();
}

double GeometryTranslator::calcCSCSpecificPhi(const TriggerPrimitive& tp) const {
  return getCSCSpecificPoint(tp).phi();
}

double GeometryTranslator::calcCSCSpecificBend(const TriggerPrimitive& tp) const { return tp.getCSCData().bend; }

bool GeometryTranslator::isCSCCounterClockwise(const std::unique_ptr<const CSCLayer>& layer) const {
  const int nStrips = layer->geometry()->numberOfStrips();
  const double phi1 = layer->centerOfStrip(1).phi();
  const double phiN = layer->centerOfStrip(nStrips).phi();
  return ((std::abs(phi1 - phiN) < M_PI && phi1 >= phiN) || (std::abs(phi1 - phiN) >= M_PI && phi1 < phiN));
}

// _____________________________________________________________________________
// DT
GlobalPoint GeometryTranslator::calcDTSpecificPoint(const TriggerPrimitive& tp) const {
  const DTChamberId baseid(tp.detId<DTChamberId>());
  // do not use this pointer for anything other than creating a trig geom
  std::unique_ptr<DTChamber> chamb(const_cast<DTChamber*>(_geodt->chamber(baseid)));
  std::unique_ptr<DTTrigGeom> trig_geom(new DTTrigGeom(chamb.get(), false));
  chamb.release();  // release it here so no one gets funny ideas
  // super layer one is the theta superlayer in a DT chamber
  // station 4 does not have a theta super layer
  // the BTI index from the theta trigger is an OR of some BTI outputs
  // so, we choose the BTI that's in the middle of the group
  // as the BTI that we get theta from
  // TODO:::::>>> need to make sure this ordering doesn't flip under wheel sign
  const int NBTI_theta = ((baseid.station() != 4) ? trig_geom->nCell(2) : trig_geom->nCell(3));
  const int bti_group = tp.getDTData().theta_bti_group;
  const unsigned bti_actual = bti_group * NBTI_theta / 7 + NBTI_theta / 14 + 1;
  DTBtiId thetaBTI;
  if (baseid.station() != 4 && bti_group != -1) {
    thetaBTI = DTBtiId(baseid, 2, bti_actual);
  } else {
    // since this is phi oriented it'll give us theta in the middle
    // of the chamber
    thetaBTI = DTBtiId(baseid, 3, 1);
  }
  const GlobalPoint& theta_gp = trig_geom->CMSPosition(thetaBTI);

  // local phi in sector -> global phi
  double phi = static_cast<double>(tp.getDTData().radialAngle) / 4096.0;  // 12 bits for 1 radian
  phi += tp.getDTData().sector * M_PI / 6.0;                              // add sector offset, sector is [0,11]

  return GlobalPoint(GlobalPoint::Polar(theta_gp.theta(), phi, theta_gp.mag()));
}

double GeometryTranslator::calcDTSpecificEta(const TriggerPrimitive& tp) const { return calcDTSpecificPoint(tp).eta(); }

double GeometryTranslator::calcDTSpecificPhi(const TriggerPrimitive& tp) const { return calcDTSpecificPoint(tp).phi(); }

// we have the bend except for station 3
double GeometryTranslator::calcDTSpecificBend(const TriggerPrimitive& tp) const {
  int bend = tp.getDTData().bendingAngle;
  double bendf = bend / 512.0;
  return bendf;
}
