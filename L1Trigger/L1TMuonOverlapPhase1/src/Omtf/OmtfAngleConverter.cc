/*
 * OmtfAngleConverter.cpp
 *
 *  Created on: Jan 14, 2019
 *      Author: kbunkow
 */

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OmtfAngleConverter.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "DataFormats/CSCDigi/interface/CSCConstants.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "L1Trigger/DTUtilities/interface/DTTrigGeom.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>

namespace {
  template <typename T>
  int sgn(T val) {
    return (T(0) < val) - (val < T(0));
  }

  //DT eta bins in the wheel +2
  const std::vector<float> bounds = {1.24, 1.14353, 1.09844, 1.05168, 1.00313, 0.952728, 0.90037, 0.8};
  //   0.8       -> 73
  //   0.85      -> 78
  //   0.9265    -> 85
  //   0.9779    -> 89.9 -> 90
  //   1.0274    -> 94.4 -> 94
  //   1.07506   -> 98.9 -> 99
  //   1.121     -> 103
  //   1.2       -> 110
  //   1.25      -> 115
  //
  // other (1.033) -> 1.033 -> 95

  int etaVal2Bit(float eta) { return bounds.rend() - std::lower_bound(bounds.rbegin(), bounds.rend(), fabs(eta)); }

  int etaVal2Code(double etaVal) {
    int sign = sgn(etaVal);
    int bit = etaVal2Bit(fabs(etaVal));
    int code = OMTFConfiguration::etaBit2Code(bit);
    return sign * code;
  }

  int etaKeyWG2Code(const CSCDetId &detId, uint16_t keyWG) {
    signed int etaCode = 121;
    if (detId.station() == 1 && detId.ring() == 2) {
      if (keyWG < 49)
        etaCode = 121;
      else if (keyWG <= 57)
        etaCode = OMTFConfiguration::etaBit2Code(0);
      else if (keyWG <= 63)
        etaCode = OMTFConfiguration::etaBit2Code(1);
    } else if (detId.station() == 1 && detId.ring() == 3) {
      if (keyWG <= 2)
        etaCode = OMTFConfiguration::etaBit2Code(2);
      else if (keyWG <= 8)
        etaCode = OMTFConfiguration::etaBit2Code(3);
      else if (keyWG <= 15)
        etaCode = OMTFConfiguration::etaBit2Code(4);
      else if (keyWG <= 23)
        etaCode = OMTFConfiguration::etaBit2Code(5);
      else if (keyWG <= 31)
        etaCode = OMTFConfiguration::etaBit2Code(6);
    } else if ((detId.station() == 2 || detId.station() == 3) && detId.ring() == 2) {
      if (keyWG < 24)
        etaCode = 121;
      else if (keyWG <= 29)
        etaCode = OMTFConfiguration::etaBit2Code(0);
      else if (keyWG <= 43)
        etaCode = OMTFConfiguration::etaBit2Code(1);
      else if (keyWG <= 49)
        etaCode = OMTFConfiguration::etaBit2Code(2);
      else if (keyWG <= 56)
        etaCode = OMTFConfiguration::etaBit2Code(3);
      else if (keyWG <= 63)
        etaCode = OMTFConfiguration::etaBit2Code(4);
    }

    if (detId.endcap() == 2)
      etaCode *= -1;
    return etaCode;
  }

}  //namespace

OmtfAngleConverter::~OmtfAngleConverter() {}

///////////////////////////////////////
///////////////////////////////////////
int OmtfAngleConverter::getGlobalEta(const DTChamberId dTChamberId,
                                     const L1MuDTChambThContainer *dtThDigis,
                                     int bxNum) const {
  //const DTChamberId dTChamberId(aDigi.whNum(),aDigi.stNum(),aDigi.scNum()+1);
  DTTrigGeom trig_geom(_geodt->chamber(dTChamberId), false);

  // super layer one is the theta superlayer in a DT chamber
  // station 4 does not have a theta super layer
  // the BTI index from the theta trigger is an OR of some BTI outputs
  // so, we choose the BTI that's in the middle of the group
  // as the BTI that we get theta from
  // TODO:::::>>> need to make sure this ordering doesn't flip under wheel sign
  const int NBTI_theta = ((dTChamberId.station() != 4) ? trig_geom.nCell(2) : trig_geom.nCell(3));

  const L1MuDTChambThDigi *theta_segm =
      dtThDigis->chThetaSegm(dTChamberId.wheel(), dTChamberId.station(), dTChamberId.sector() - 1, bxNum);

  int bti_group = -1;
  if (theta_segm) {
    for (unsigned int i = 0; i < 7; ++i)
      if (theta_segm->position(i) && bti_group < 0)
        bti_group = i;
      else if (theta_segm->position(i) && bti_group > -1)
        bti_group = 511;
  }

  //TODO why bti_group == 511 (meaning there is more then one bit fired) is converted into 95, but not into middle of the chamber?
  int iEta = 0;
  if (bti_group == 511)
    iEta = 95;
  else if (bti_group == -1 && dTChamberId.station() == 1)
    iEta = 92;
  else if (bti_group == -1 && dTChamberId.station() == 2)
    iEta = 79;
  else if (bti_group == -1 && dTChamberId.station() == 3)
    iEta = 75;
  else if (dTChamberId.station() != 4 && bti_group >= 0) {
    unsigned bti_actual = bti_group * NBTI_theta / 7 + NBTI_theta / 14 + 1;
    DTBtiId thetaBTI = DTBtiId(dTChamberId, 2, bti_actual);
    GlobalPoint theta_gp = trig_geom.CMSPosition(thetaBTI);
    iEta = etaVal2Code(fabs(theta_gp.eta()));
  }
  int signEta = sgn(dTChamberId.wheel());
  iEta *= signEta;

  if (config->getStubEtaEncoding() == ProcConfigurationBase::StubEtaEncoding::bits)
    return OMTFConfiguration::eta2Bits(abs(iEta));
  else if (config->getStubEtaEncoding() == ProcConfigurationBase::StubEtaEncoding::valueP1Scale)
    return abs(iEta);

  return 0;
}

///////////////////////////////////////
///////////////////////////////////////
int OmtfAngleConverter::getGlobalEta(unsigned int rawid, const CSCCorrelatedLCTDigi &aDigi, float &r) const {
  ///Code taken from GeometryTranslator.
  ///Will be replaced by direct CSC phi local to global scale
  ///transformation as used in FPGA implementation

  // a lot of this is transcription and consolidation of the CSC
  // global phi calculation code
  // this works directly with the geometry
  // rather than using the old phi luts
  const CSCDetId id(rawid);
  // we should change this to weak_ptrs at some point
  // requires introducing std::shared_ptrs to geometry
  auto chamb = _geocsc->chamber(id);
  auto layer_geom = chamb->layer(CSCConstants::KEY_ALCT_LAYER)->geometry();
  auto layer = chamb->layer(CSCConstants::KEY_ALCT_LAYER);

  const uint16_t halfstrip = aDigi.getStrip();
  //const uint16_t pattern = aDigi.getPattern();
  const uint16_t keyWG = aDigi.getKeyWG();
  //const unsigned maxStrips = layer_geom->numberOfStrips();

  // so we can extend this later
  // assume TMB2007 half-strips only as baseline
  double offset = 0.0;
  //K.B. CSCPatternLUT is removed since CMSSW_11_2_1, but it looks that this offset is effectively not needed here
  //offset = CSCPatternLUT::get2007Position(pattern);

  const unsigned halfstrip_offs = unsigned(0.5 + halfstrip + offset);
  const unsigned strip = halfstrip_offs / 2 + 1;  // geom starts from 1

  // the rough location of the hit at the ALCT key layer
  // we will refine this using the half strip information
  const LocalPoint coarse_lp = layer_geom->stripWireGroupIntersection(strip, keyWG);
  const GlobalPoint coarse_gp = layer->surface().toGlobal(coarse_lp);

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

  //TODO for phase 2, add firmware like, fixed point conversion from keyWG to eta and r
  const LocalPoint lpWg = layer_geom->localCenterOfWireGroup(keyWG);
  const GlobalPoint gpWg = layer->surface().toGlobal(lpWg);
  r = round(gpWg.perp());

  if (config->getStubEtaEncoding() == ProcConfigurationBase::StubEtaEncoding::bits)
    return OMTFConfiguration::eta2Bits(abs(etaKeyWG2Code(id, keyWG)));
  else if (config->getStubEtaEncoding() == ProcConfigurationBase::StubEtaEncoding::valueP1Scale) {
    return config->etaToHwEta(abs(gpWg.eta()));
  } else {
    return 0;
  }
}
///////////////////////////////////////
///////////////////////////////////////
int OmtfAngleConverter::getGlobalEtaRpc(unsigned int rawid, const unsigned int &strip, float &r) const {
  const RPCDetId id(rawid);
  auto roll = _georpc->roll(id);
  const LocalPoint lp = roll->centreOfStrip((int)strip);
  const GlobalPoint gp = roll->toGlobal(lp);

  if (id.region() != 0) {  //outside barrel
    r = gp.perp();
  }

  if (config->getStubEtaEncoding() == ProcConfigurationBase::StubEtaEncoding::bits)
    return OMTFConfiguration::eta2Bits(abs(etaVal2Code(gp.eta())));
  else if (config->getStubEtaEncoding() == ProcConfigurationBase::StubEtaEncoding::valueP1Scale)
    return abs(config->etaToHwEta((gp.eta())));

  return 0;
}

///////////////////////////////////////
///////////////////////////////////////
