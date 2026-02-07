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

  int fixCscOffsetGeom(int offsetLoc) {
    // fix for CSC geo dependence from GlobalTag

    // dump of CSC offsets for MC global tag
    const std::vector<int> offCSC = {-154, -133, -17, -4,  4,   17,  133, 146, 154, 167, 283, 296, 304, 317,
                                     433,  446,  454, 467, 583, 596, 604, 617, 733, 746, 754, 767, 883, 904};
    auto gep = std::lower_bound(offCSC.begin(), offCSC.end(), offsetLoc);
    int fixOff = (gep != offCSC.end()) ? *gep : *(gep - 1);
    if (gep != offCSC.begin() && std::abs(*(gep - 1) - offsetLoc) < std::abs(fixOff - offsetLoc))
      fixOff = *(gep - 1);
    return fixOff;
  }

  //phase-1: DT eta bins in the wheel +2
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

  int etaKeyWG2Code(const CSCDetId& detId, uint16_t keyWG) {
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
void OmtfAngleConverter::checkAndUpdateGeometry(const edm::EventSetup& es,
                                                const ProcConfigurationBase* config,
                                                const MuonGeometryTokens& muonGeometryTokens) {
  if (muonGeometryRecordWatcher.check(es)) {
    _georpc = es.getHandle(muonGeometryTokens.rpcGeometryEsToken);
    _geocsc = es.getHandle(muonGeometryTokens.cscGeometryEsToken);
    _geodt = es.getHandle(muonGeometryTokens.dtGeometryEsToken);
  }
  this->config = config;
  nPhiBins = config->nPhiBins();
}

///////////////////////////////////////
///////////////////////////////////////
int OmtfAngleConverter::getProcessorPhi(int phiZero, l1t::tftype part, int dtScNum, int dtPhi) const {
  int dtPhiBins = 4096;

  double hsPhiPitch = 2 * M_PI / nPhiBins;  // width of phi Pitch, related to halfStrip at CSC station 2

  int sector = dtScNum + 1;  //NOTE: there is a inconsistency in DT sector numb. Thus +1 needed to get detector numb.

  double scale = 1. / dtPhiBins / hsPhiPitch;
  int scale_coeff = lround(scale * pow(2, 11));  // 216.2688

  int ichamber = sector - 1;
  if (ichamber > 6)
    ichamber = ichamber - 12;

  int offsetGlobal = (int)nPhiBins * ichamber / 12;

  int phiConverted = floor(dtPhi * scale_coeff / pow(2, 11)) + offsetGlobal - phiZero;

  //LogTrace("l1tOmtfEventPrint")<<__FUNCTION__<<":"<<__LINE__<<" phiZero "<<phiZero<<" phiDT "<<phiDT<<" sector "<<sector<<" ichamber "<<ichamber<<" offsetGlobal "<<offsetGlobal<<" phi "<<phi<<" foldPhi(phi) "<<omtfConfig->foldPhi(phi)<<std::endl;
  return config->foldPhi(phiConverted);
}
///////////////////////////////////////
///////////////////////////////////////
int OmtfAngleConverter::getProcessorPhi(
    int phiZero, l1t::tftype part, const CSCDetId& csc, const CSCCorrelatedLCTDigi& digi, unsigned int iInput) const {
  const double hsPhiPitch = 2 * M_PI / nPhiBins;
  //
  // get offset for each chamber.
  // FIXME: These parameters depends on processor and chamber only so may be precomputed and put in map
  //

  int halfStrip = digi.getStrip();  // returns halfStrip 0..159

  const CSCChamber* chamber = _geocsc->chamber(csc);

  //in the PhaseIITDRSpring19DR dataset (generated with CMSSW_10_6_1_patch2?), in case of the ME1/1 ring 4 (higher eta) the detId in the CSCCorrelatedLCTDigiCollection is ME1/1 ring 1 (instead ME1/1/4 as it was before),
  //and the digi.getStrip() is increased by 2*64 (i.e. number of half strips in the chamber roll)
  if (csc.station() == 1 && csc.ring() == 1 && halfStrip > 128) {
    CSCDetId cscME11 = CSCDetId(csc.endcap(), csc.station(), 4, csc.chamber());  //changing ring  to 4
    chamber = _geocsc->chamber(cscME11);
  }

  const CSCChamberSpecs* cspec = chamber->specs();
  const CSCLayer* layer = chamber->layer(3);
  int order = (layer->centerOfStrip(2).phi() - layer->centerOfStrip(1).phi() > 0) ? 1 : -1;
  double stripPhiPitch = cspec->stripPhiPitch();
  double scale = std::abs(stripPhiPitch / hsPhiPitch / 2.);
  if (std::abs(scale - 1.) < 0.0002)
    scale = 1.;

  double phiHalfStrip0 = layer->centerOfStrip(1).phi() - order * stripPhiPitch / 4.;

  int offsetLoc = lround((phiHalfStrip0) / hsPhiPitch - phiZero);
  offsetLoc = config->foldPhi(offsetLoc);

  if (csc.station() == 1 && csc.ring() == 1 && halfStrip > 128) {  //ME1/1/
                                                                   /*    if(cspec->nStrips() != 64)
      edm::LogImportant("l1tOmtfEventPrint") <<__FUNCTION__<<":"<<__LINE__<<" cspec->nStrips() != 64 in case of the ME1/1, phi of the muon stub will be not correct. chamber "
      <<csc<<" cspec->nStrips() "<<cspec->nStrips()<<std::endl;
      this checks has no sense - the ME1/1/ ring 4 has cspec->nStrips() = 48. but the offset of 128 half strips in the digi.getStrip() looks to be good*/
    halfStrip -= 128;
  }

  //FIXME: to be checked (only important for ME1/3) keep more bits for offset, truncate at the end

  int fixOff = offsetLoc;
  // a quick fix for towards geometry changes due to global tag.
  // in case of MC tag fixOff should be identical to offsetLoc

  if (config->getFixCscGeometryOffset()) {
    if (config->nProcessors() == 6)          //phase1
      fixOff = fixCscOffsetGeom(offsetLoc);  //TODO does not work in when phiZero is always 0. Fix this
    else if (config->nProcessors() == 3) {   //phase2
      //TODO fix this bricolage!!!!!!!!!!!!!!
      if (iInput >= 14)
        fixOff = fixCscOffsetGeom(offsetLoc - 900) + 900;
      else
        fixOff = fixCscOffsetGeom(offsetLoc);
    }
  }
  int phi = fixOff + order * scale * halfStrip;
  //the phi conversion is done like above - and not simply converting the layer->centerOfStrip(halfStrip/2 +1).phi() - to mimic this what is done by the firmware,
  //where phi of the stub is calculated with use of the offset and scale provided by an register

  /*//debug
  auto localPoint = layer->toLocal(layer->centerOfStrip(halfStrip));
  LogTrace("l1tOmtfEventPrint") << __FUNCTION__ << ":" << 147 << " csc: " <<csc.rawId()<<" "<< csc<<" layer "<<layer->id()<<" "<<layer->id().rawId()
      << " halfStrip "<<halfStrip<<" phiGlobal " << layer->centerOfStrip(halfStrip).phi()<<" local phi "<<localPoint.phi()<<" x "<<localPoint.x()<<" y "<<localPoint.y() <<std::endl;
  */

  /*//debug
  auto radToDeg = [](double rad) { return (180. / M_PI * rad); };
  LogTrace("l1tOmtfEventPrint") <<__FUNCTION__<<":"<<__LINE__<<" "<<std::setw(16)<<csc<<" phiZero "<<phiZero<<" hs: "<<std::setw(3)<< halfStrip <<" phiHalfStrip0 "<<std::setw(10)<<radToDeg(phiHalfStrip0)<<" offset: " << offsetLoc
      <<" oder*scale: "<<std::setw(10)<< order*scale
       <<" phi: " <<phi<<" foldPhi(phi) "<<config->foldPhi(phi)<<" ("<<offsetLoc + order*scale*halfStrip<<")"<<" centerOfStrip "<<std::setw(10)<< radToDeg( layer->centerOfStrip(halfStrip/2 +1).phi() )
       <<" centerOfStrip/hsPhiPitch "<< ( (layer->centerOfStrip(halfStrip/2 + 1).phi() )/hsPhiPitch)<<"  hsPhiPitch "<<hsPhiPitch
       //<<" phiSpan.f "<<layer->geometry()->phiSpan().first<<" phiSpan.s "<<layer->geometry()->phiSpan().second
       <<" nStrips "<<cspec->nStrips()
       //<<" strip 1 "<<radToDeg( layer->centerOfStrip(1).phi() )<<" strip last "<<radToDeg( layer->centerOfStrip(cspec->nStrips()).phi() )
       << std::endl;*/

  return config->foldPhi(phi);
}

///////////////////////////////////////
///////////////////////////////////////
int OmtfAngleConverter::getProcessorPhi(
    int phiZero, l1t::tftype part, const RPCDetId& rollId, const unsigned int& digi1, const unsigned int& digi2) const {
  const double hsPhiPitch = 2 * M_PI / nPhiBins;
  const int dummy = nPhiBins;
  const RPCRoll* roll = _georpc->roll(rollId);
  if (!roll)
    return dummy;

  double stripPhi1 = (roll->toGlobal(roll->centreOfStrip((int)digi1))).phi();  // note [-pi,pi]
  double stripPhi2 = (roll->toGlobal(roll->centreOfStrip((int)digi2))).phi();  // note [-pi,pi]

  // the case when the two strips are on different sides of phi = pi
  if (std::signbit(stripPhi1) != std::signbit(stripPhi2) && std::abs(stripPhi1) > M_PI / 2.) {
    if (std::signbit(stripPhi1)) {  //stripPhi1 is negative
      stripPhi1 += 2 * M_PI;
    } else  //stripPhi2 is negative
      stripPhi2 += 2 * M_PI;
  }
  int halfStrip = lround(((stripPhi1 + stripPhi2) / 2.) / hsPhiPitch);
  halfStrip = config->foldPhi(halfStrip);  //only for the case when the two strips are on different sides of phi = pi

  LogTrace("l1tOmtfEventPrint") << __FUNCTION__ << ":" << 185 << " roll " << rollId.rawId() << " " << rollId
                                << " cluster: firstStrip " << digi1 << " stripPhi1Global " << stripPhi1
                                << " stripPhi1LocalPhi " << roll->centreOfStrip((int)digi1).x() << " y "
                                << roll->centreOfStrip((int)digi1).y() << " lastStrip " << digi2 << " stripPhi2Global "
                                << stripPhi2 << " stripPhi2LocalPhi x " << roll->centreOfStrip((int)digi2).x() << " y "
                                << roll->centreOfStrip((int)digi2).y() << " halfStrip " << halfStrip << std::endl;

  return config->foldPhi(halfStrip - phiZero);
}

///////////////////////////////////////
///////////////////////////////////////
int OmtfAngleConverter::getGlobalEta(const DTChamberId dTChamberId,
                                     const L1MuDTChambThContainer* dtThDigis,
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

  const L1MuDTChambThDigi* theta_segm =
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
  else if (config->getStubEtaEncoding() >= ProcConfigurationBase::StubEtaEncoding::valueP1Scale)
    return abs(iEta);

  return 0;
}

///////////////////////////////////////
///////////////////////////////////////
int OmtfAngleConverter::getGlobalEta(unsigned int rawid, const CSCCorrelatedLCTDigi& aDigi, float& r) const {
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
  else if (config->getStubEtaEncoding() >= ProcConfigurationBase::StubEtaEncoding::valueP1Scale) {
    return config->etaToHwEta(abs(gpWg.eta()));
  } else {
    return 0;
  }
}
///////////////////////////////////////
///////////////////////////////////////
int OmtfAngleConverter::getGlobalEtaRpc(unsigned int rawid, const unsigned int& strip, float& r) const {
  const RPCDetId id(rawid);
  auto roll = _georpc->roll(id);
  const LocalPoint lp = roll->centreOfStrip((int)strip);
  const GlobalPoint gp = roll->toGlobal(lp);

  if (id.region() != 0) {  //outside barrel
    r = gp.perp();
  }

  if (config->getStubEtaEncoding() == ProcConfigurationBase::StubEtaEncoding::bits)
    return OMTFConfiguration::eta2Bits(abs(etaVal2Code(gp.eta())));
  else if (config->getStubEtaEncoding() >= ProcConfigurationBase::StubEtaEncoding::valueP1Scale)
    return abs(config->etaToHwEta((gp.eta())));

  return 0;
}

///////////////////////////////////////
///////////////////////////////////////
bool OmtfAngleConverter::isCSCCounterClockwise(const CSCLayer* layer) const {
  const int nStrips = layer->geometry()->numberOfStrips();
  const double phi1 = layer->centerOfStrip(1).phi();
  const double phiN = layer->centerOfStrip(nStrips).phi();
  return ((std::abs(phi1 - phiN) < M_PI && phi1 >= phiN) || (std::abs(phi1 - phiN) >= M_PI && phi1 < phiN));
}
///////////////////////////////////////
///////////////////////////////////////
const int OmtfAngleConverter::findBTIgroup(const L1MuDTChambPhDigi& aDigi, const L1MuDTChambThContainer* dtThDigis) {
  int bti_group = -1;

  const L1MuDTChambThDigi* theta_segm =
      dtThDigis->chThetaSegm(aDigi.whNum(), aDigi.stNum(), aDigi.scNum(), aDigi.bxNum());
  if (!theta_segm)
    return bti_group;

  for (unsigned int i = 0; i < 7; ++i) {
    if (theta_segm->position(i) && bti_group < 0)
      bti_group = i;
    ///If there are more than one theta digi we do not take is
    ///due to unresolved ambiguity. In this case we take eta of the
    ///middle of the chamber.
    else if (theta_segm->position(i) && bti_group > -1)
      return -1;
  }

  return bti_group;
}
///////////////////////////////////////
///////////////////////////////////////
