#include "L1Trigger/L1TMuonOverlapPhase1/interface/AngleConverterBase.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"

#include "DataFormats/CSCDigi/interface/CSCConstants.h"
#include "L1Trigger/DTUtilities/interface/DTTrigGeom.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
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

}  // namespace

AngleConverterBase::AngleConverterBase() : _geom_cache_id(0ULL) {}
///////////////////////////////////////
///////////////////////////////////////
AngleConverterBase::~AngleConverterBase() {}
///////////////////////////////////////
///////////////////////////////////////
void AngleConverterBase::checkAndUpdateGeometry(const edm::EventSetup& es,
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
int AngleConverterBase::getProcessorPhi(int phiZero, l1t::tftype part, int dtScNum, int dtPhi) const {
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
int AngleConverterBase::getProcessorPhi(
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
int AngleConverterBase::getProcessorPhi(
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

int AngleConverterBase::getProcessorPhi(unsigned int iProcessor,
                                        l1t::tftype part,
                                        const RPCDetId& rollId,
                                        const unsigned int& digi) const {
  const double hsPhiPitch = 2 * M_PI / nPhiBins;
  const int dummy = nPhiBins;
  int processor = iProcessor + 1;
  const RPCRoll* roll = _georpc->roll(rollId);
  if (!roll)
    return dummy;

  double phi15deg = M_PI / 3. * (processor - 1) + M_PI / 12.;
  // "0" is 15degree moved cyclically to each processor, note [0,2pi]

  double stripPhi = (roll->toGlobal(roll->centreOfStrip((int)digi))).phi();  // note [-pi,pi]

  // adjust [0,2pi] and [-pi,pi] to get deltaPhi difference properly
  switch (processor) {
    case 1:
      break;
    case 6: {
      phi15deg -= 2 * M_PI;
      break;
    }
    default: {
      if (stripPhi < 0)
        stripPhi += 2 * M_PI;
      break;
    }
  }

  // local angle in CSC halfStrip usnits
  int halfStrip = lround((stripPhi - phi15deg) / hsPhiPitch);

  return halfStrip;
}
///////////////////////////////////////
///////////////////////////////////////
EtaValue AngleConverterBase::getGlobalEtaDt(const DTChamberId& detId) const {
  Local2DPoint chamberMiddleLP(0, 0);
  GlobalPoint chamberMiddleGP = _geodt->chamber(detId)->toGlobal(chamberMiddleLP);

  const DTChamberId baseidNeigh(detId.wheel() + (detId.wheel() >= 0 ? -1 : +1), detId.station(), detId.sector());
  GlobalPoint chambNeighMiddleGP = _geodt->chamber(baseidNeigh)->toGlobal(chamberMiddleLP);

  EtaValue etaValue = {
      config->etaToHwEta(chamberMiddleGP.eta()),
      config->etaToHwEta(std::abs(chamberMiddleGP.eta() - chambNeighMiddleGP.eta())) / 2,
      0,  //quality
      0,  //bx
      0   //timin
  };

  //LogTrace("l1tOmtfEventPrint")<<__FUNCTION__<<":"<<__LINE__<<" rawid "<<detId.rawId()<<" baseid "<<detId<<" chamberMiddleGP.eta() "<<chamberMiddleGP.eta()<<" eta "<<etaValue.eta<<" etaSigma "<<etaValue.etaSigma<<std::endl;
  return etaValue;
}

///////////////////////////////////////
///////////////////////////////////////
void AngleConverterBase::getGlobalEta(const L1MuDTChambThDigi& thetaDigi, std::vector<EtaValue>& etaSegments) const {
  const DTChamberId baseid(thetaDigi.whNum(), thetaDigi.stNum(), thetaDigi.scNum() + 1);
  DTTrigGeom trig_geom(_geodt->chamber(baseid), false);

  // super layer 2 is the theta superlayer in a DT chamber
  // station 4 does not have a theta super layer
  // the BTI index from the theta trigger is an OR of some BTI outputs
  // so, we choose the BTI that's in the middle of the group
  // as the BTI that we get theta from
  // TODO:::::>>> need to make sure this ordering doesn't flip under wheel sign
  const int NBTI_theta = trig_geom.nCell(2);
  for (unsigned int btiGroup = 0; btiGroup < 7; ++btiGroup) {
    if (thetaDigi.position(btiGroup)) {
      unsigned btiActual = btiGroup * NBTI_theta / 7 + NBTI_theta / 14 + 1;
      DTBtiId thetaBTI = DTBtiId(baseid, 2, btiActual);
      GlobalPoint theta_gp = trig_geom.CMSPosition(thetaBTI);

      EtaValue etaValue = {
          config->etaToHwEta(theta_gp.eta()),
          0,
          thetaDigi.quality(btiGroup),
          thetaDigi.bxNum(),
          0  //TODO what about sub-bx timing???
      };
      etaSegments.emplace_back(etaValue);

      //LogTrace("l1tOmtfEventPrint")<<__FUNCTION__<<":"<<__LINE__<<" bx "<<thetaDigi.bxNum()<<" baseid "<<baseid<<" btiGroup "<<btiGroup<<" quality "<<thetaDigi.quality(btiGroup)<<" theta_gp.eta() "<<theta_gp.eta()<<" eta "<<etaValue.eta<<" etaSigma "<<etaValue.etaSigma<<std::endl;
    }
  }
}

std::vector<EtaValue> AngleConverterBase::getGlobalEta(const L1MuDTChambThContainer* dtThDigis,
                                                       int bxFrom,
                                                       int bxTo) const {
  //LogTrace("l1tOmtfEventPrint")<<__FUNCTION__<<":"<<__LINE__<<" dtThDigis size "<<dtThDigis->getContainer()->size()<<std::endl;

  std::vector<EtaValue> etaSegments;

  for (auto& thetaDigi : (*(dtThDigis->getContainer()))) {
    if (thetaDigi.bxNum() >= bxFrom && thetaDigi.bxNum() <= bxTo) {
      getGlobalEta(thetaDigi, etaSegments);
    }
  }
  return etaSegments;
}

//just read from the drawing
float AngleConverterBase::cscChamberEtaSize(const CSCDetId& detId) const {
  if (detId.station() == 1) {
    if (detId.ring() == 1)
      return (2.5 - 1.6) / 2.;
    ///ME1/1 lower eta (b?, eta < ~2.1), L1TkMuonBayes eta bins 6-11 - but getGlobalEtaCsc(const CSCDetId& detId) gives the midle of the full chamber, so here we put the size of the full chamber
    if (detId.ring() == 2)
      return (1.7 - 1.2) / 2.;
    if (detId.ring() == 3)
      return (1.12 - 0.9) / 2.;
    if (detId.ring() == 4)
      return (2.5 - 1.6) / 2.;  ///ME1/1 higher eta (a?, eta > ~2.1), L1TkMuonBayes eta bins 10-15
  } else if (detId.station() == 2) {
    if (detId.ring() == 1)
      return (2.5 - 1.6) / 2.;
    if (detId.ring() == 2)
      return (1.6 - 1.0) / 2.;
  } else if (detId.station() == 3) {
    if (detId.ring() == 1)
      return (2.5 - 1.7) / 2.;
    if (detId.ring() == 2)
      return (1.7 - 1.1) / 2.;
  } else if (detId.station() == 4) {
    if (detId.ring() == 1)
      return (2.45 - 1.8) / 2.;
    if (detId.ring() == 2)
      return (1.8 - 1.2) / 2.;
  }
  return 0;
}

EtaValue AngleConverterBase::getGlobalEta(const CSCDetId& detId, const CSCCorrelatedLCTDigi& aDigi) const {
  ///Code taken from GeometryTranslator.
  ///Will be replaced by direct CSC phi local to global scale
  ///transformation as used in FPGA implementation

  // alot of this is transcription and consolidation of the CSC
  // global phi calculation code
  // this works directly with the geometry
  // rather than using the old phi luts

  auto chamb = _geocsc->chamber(detId);
  auto layer_geom = chamb->layer(CSCConstants::KEY_ALCT_LAYER)->geometry();
  auto layer = chamb->layer(CSCConstants::KEY_ALCT_LAYER);

  const uint16_t keyWG = aDigi.getKeyWG();

  const LocalPoint lpWg = layer_geom->localCenterOfWireGroup(keyWG);
  const GlobalPoint gpWg = layer->surface().toGlobal(lpWg);

  EtaValue etaSegment = {
      config->etaToHwEta(gpWg.eta()),
      0,  //config->etaToHwEta(cscChamberEtaSize(id) ),
      0,
      aDigi.getBX(),
      0  //tming???
  };

  //LogTrace("l1tOmtfEventPrint")<<__FUNCTION__<<":"<<__LINE__<<" csc "<<detId<<" eta "<<gpWg.eta()<<" etaHw "<<etaSegment.eta<<" etaSigma "<<etaSegment.etaSigma<<std::endl;
  return etaSegment;
}

//TODO the CSC ME1/1 has strips divided in two parts: a and b, so this function in principle can include that,
//then it should also receive the roll number as parameter, off course implementation should be different then
EtaValue AngleConverterBase::getGlobalEtaCsc(const CSCDetId& detId) const {
  auto chamb = _geocsc->chamber(detId);

  Local2DPoint chamberMiddleLP(0, 0);
  GlobalPoint chamberMiddleGP = chamb->toGlobal(chamberMiddleLP);

  EtaValue etaValue = {
      config->etaToHwEta(chamberMiddleGP.eta()),
      config->etaToHwEta(cscChamberEtaSize(detId)),
      0,
      0,  //bx
      0   //timnig
  };

  //LogTrace("l1tOmtfEventPrint")<<__FUNCTION__<<":"<<__LINE__<<" rawid "<<detId.rawId()<<" detId "<<detId<<" chamberMiddleGP.eta() "<<chamberMiddleGP.eta()<<" eta "<<etaValue.eta<<" etaSigma "<<etaValue.etaSigma<<std::endl;
  return etaValue;
}

///////////////////////////////////////
///////////////////////////////////////
EtaValue AngleConverterBase::getGlobalEta(unsigned int rawid, const unsigned int& strip) const {
  const RPCDetId id(rawid);

  auto roll = _georpc->roll(id);
  const LocalPoint lp = roll->centreOfStrip((int)strip);
  const GlobalPoint gp = roll->toGlobal(lp);

  int neighbRoll = 1;  //neighbor roll in eta
  //roll->chamber()->nrolls() does not work
  if (id.region() == 0) {  //barel
    if (id.station() == 2 && ((std::abs(id.ring()) == 2 && id.layer() == 2) ||
                              (std::abs(id.ring()) != 2 && id.layer() == 1))) {  //three-roll chamber
      if (id.roll() == 2)
        neighbRoll = 1;
      else {
        neighbRoll = 2;
      }
    } else  //two-roll chamber
      neighbRoll = (id.roll() == 1 ? 3 : 1);
  } else {  //endcap
    neighbRoll = id.roll() + (id.roll() == 1 ? +1 : -1);
  }

  const RPCDetId idNeigh =
      RPCDetId(id.region(), id.ring(), id.station(), id.sector(), id.layer(), id.subsector(), neighbRoll);
  //LogTrace("l1tOmtfEventPrint")<<__FUNCTION__<<":"<<__LINE__<<" rpc "<<id<<std::endl;
  //LogTrace("l1tOmtfEventPrint")<<__FUNCTION__<<":"<<__LINE__<<" rpc "<<idNeigh<<std::endl;

  auto rollNeigh = _georpc->roll(idNeigh);
  const LocalPoint lpNeigh = rollNeigh->centreOfStrip((int)strip);
  const GlobalPoint gpNeigh = rollNeigh->toGlobal(lpNeigh);

  EtaValue etaValue = {config->etaToHwEta(gp.eta()),
                       config->etaToHwEta(std::abs(gp.eta() - gpNeigh.eta())) /
                           2,  //half of the size of the strip in eta - not precise, but OK
                       0};

  //LogTrace("l1tOmtfEventPrint")<<__FUNCTION__<<":"<<__LINE__<<" rpc "<<id<<" eta "<<gp.eta()<<" etaHw "<<etaValue.eta<<" etaSigma "<<etaValue.etaSigma<<std::endl;
  return etaValue;
}
///////////////////////////////////////
///////////////////////////////////////
bool AngleConverterBase::isCSCCounterClockwise(const CSCLayer* layer) const {
  const int nStrips = layer->geometry()->numberOfStrips();
  const double phi1 = layer->centerOfStrip(1).phi();
  const double phiN = layer->centerOfStrip(nStrips).phi();
  return ((std::abs(phi1 - phiN) < M_PI && phi1 >= phiN) || (std::abs(phi1 - phiN) >= M_PI && phi1 < phiN));
}
///////////////////////////////////////
///////////////////////////////////////
const int AngleConverterBase::findBTIgroup(const L1MuDTChambPhDigi& aDigi, const L1MuDTChambThContainer* dtThDigis) {
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
