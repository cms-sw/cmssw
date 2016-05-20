#include "L1Trigger/L1TMuonOverlap/interface/AngleConverter.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCPatternLUT.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "L1Trigger/DTUtilities/interface/DTTrigGeom.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"

#include <cmath> 

namespace {
template <typename T> int sgn(T val) { return (T(0) < val) - (val < T(0)); }

std::vector<float> bounds = { 1.27, 1.14353, 1.09844, 1.05168, 1.00313, 0.952728, 0.90037, 0.8};
//   0.8       -> 73
//   0.85      -> 78
//   0.9265    -> 85
//   0.9779    -> 89.9 -> 90
//   1.0274    -> 94.4 -> 94
//   1.07506   -> 98.9 -> 99
//   1.121     -> 103
//   1.2       -> 110
//   1.26      -> 115
//
// other (1.35) -> 1.035 -> 95

int etaVal2Bit(float eta) { return bounds.rend() - std::lower_bound (bounds.rbegin(), bounds.rend(), fabs(eta) ); }

int etaBit2Code( unsigned int bit) {
  int code = 73;
  switch (bit) {
    case 0 : {code = 115; break;}
    case 1 : {code = 110; break; }
    case 2 : {code = 103; break; }
    case 3 : {code = 99; break; }
    case 4 : {code = 94; break; }
    case 5 : {code = 90; break; }
    case 6 : {code = 85; break; }
    case 7 : {code = 78; break; }
    case 8 : {code = 73; break; }
    default: {code = 95; break; }
  }
  return code;
}

int etaVal2Code( double etaVal) {
  int sign = sgn(etaVal);
  int bit = etaVal2Bit( fabs(etaVal) );
  int code = etaBit2Code(bit); 
  return sign*code;
}

int etaKeyWG2Code(const CSCDetId& detId, uint16_t keyWG) {
  unsigned int etaCode = 0;
  if (detId.station()==1 && detId.ring()==2) {
    if (keyWG <58)       etaCode = etaBit2Code(0);
    else if (keyWG <=63) etaCode = etaBit2Code(1);
  } 
  else if (detId.station()==1 && detId.ring()==3) {
    if (keyWG <= 2)       etaCode = etaBit2Code(2);
    else if (keyWG <=  8) etaCode = etaBit2Code(3);
    else if (keyWG <= 15) etaCode = etaBit2Code(4);
    else if (keyWG <= 23) etaCode = etaBit2Code(5);
    else if (keyWG <= 31) etaCode = etaBit2Code(6);
  } 
  else if ( (detId.station()==2 || detId.station()==3) && detId.ring()==2) {
    if (keyWG <= 29)       etaCode = etaBit2Code(0);
    else if (keyWG <=  45) etaCode = etaBit2Code(1);
    else if (keyWG <=  52) etaCode = etaBit2Code(2);
    else if (keyWG <=  60) etaCode = etaBit2Code(3);
    else if (keyWG <=  63) etaCode = etaBit2Code(4);
  }
  if (detId.endcap()==2) etaCode *= -1;
  return etaCode;
}

}


AngleConverter::AngleConverter(): _geom_cache_id(0ULL) { }
///////////////////////////////////////
///////////////////////////////////////
AngleConverter::~AngleConverter() {  }
///////////////////////////////////////
///////////////////////////////////////
void AngleConverter::checkAndUpdateGeometry(const edm::EventSetup& es, unsigned int phiBins) {
  const MuonGeometryRecord& geom = es.get<MuonGeometryRecord>();
  unsigned long long geomid = geom.cacheIdentifier();
  if( _geom_cache_id != geomid ) {
    geom.get(_georpc);  
    geom.get(_geocsc);    
    geom.get(_geodt);
    _geom_cache_id = geomid;
  }

  nPhiBins = phiBins;

}
///////////////////////////////////////
///////////////////////////////////////
int AngleConverter::getProcessorPhi(unsigned int iProcessor, l1t::tftype part, const L1MuDTChambPhDigi &digi) const
{

  double hsPhiPitch = 2*M_PI/nPhiBins; // width of phi Pitch, related to halfStrip at CSC station 2
  const int dummy = nPhiBins;
  int processor= iProcessor+1;                           // FIXME: get from OMTF name when available
  int posneg = (part==l1t::tftype::omtf_pos) ? 1 : -1;        // FIXME: get from OMTF name

  int sector  = digi.scNum()+1;   //NOTE: there is a inconsistency in DT sector numb. Thus +1 needed to get detector numb.
  int wheel   = digi.whNum();
  int station = digi.stNum();
  int phiDT   = digi.phi();

  if (posneg*2 != wheel) return dummy;
  if (station > 3 ) return dummy;

  //FIXME: update the following two lines with proper method when Connections introduced
  if (processor !=6 && !(sector >= processor*2-1 && sector <= processor*2+1) ) return dummy;
  if (processor ==6 && !(sector >= 11 || sector==1) ) return dummy;

  // ichamber is consecutive chamber connected to processor, starting from 0 (overlaping one)
  int ichamber = sector-1-2*(processor-1);
  if (ichamber < 0) ichamber += 12;

  int offsetLoc = lround( ((ichamber-1)*M_PI/6+M_PI/12.)/hsPhiPitch );
  double scale = 1./4096/hsPhiPitch;

  int phi = static_cast<int>(phiDT*scale) + offsetLoc;

  return phi;
}
///////////////////////////////////////
///////////////////////////////////////
int AngleConverter::getProcessorPhi(unsigned int iProcessor, l1t::tftype part, const CSCDetId & csc, const CSCCorrelatedLCTDigi &digi) const
{

  const double hsPhiPitch = 2*M_PI/nPhiBins;
  const int dummy = nPhiBins;
  int processor= iProcessor+1;                           // FIXME: get from OMTF name when available
  int posneg = (part==l1t::tftype::omtf_pos) ? 1 : -1;        // FIXME: get from OMTF name



  // filter out chambers not connected to OMTF board
  // FIXME: temporary - use Connections or relay that filtering done before.
  if (posneg != csc.zendcap()) return dummy;
  if ( csc.ring() != 3 && !(csc.ring()==2 && (csc.station()==2 || csc.station()==3 || csc.station()==1)) ) return dummy;
  if (processor !=6) {
    if (csc.chamber() < (processor-1)*6 + 2) return dummy;
    if (csc.chamber() > (processor-1)*6 + 8) return dummy;
  } else {
    if (csc.chamber() > 2 && csc.chamber() < 32) return dummy;
  }

  //
  // assign number 0..6, consecutive processor for a processor
  //
  //int ichamber = (csc.chamber()-2-6*(processor-1));
  //if (ichamber < 0) ichamber += 36;

  //
  // get offset for each chamber.
  // FIXME: These parameters depends on processor and chamber only so may be precomputed and put in map
  //
  const CSCChamber* chamber = _geocsc->chamber(csc);
  const CSCChamberSpecs* cspec = chamber->specs();
  const CSCLayer* layer = chamber->layer(3);
  int order = ( layer->centerOfStrip(2).phi() - layer->centerOfStrip(1).phi()> 0) ? 1 : -1;
  double stripPhiPitch = cspec->stripPhiPitch();
  double scale = fabs(stripPhiPitch/hsPhiPitch/2.); if ( fabs(scale-1.) < 0.0002) scale=1.;
  double phi15deg = M_PI/3.*(processor-1)+M_PI/12.;
  double phiHalfStrip0 = layer->centerOfStrip(10).phi() - order*9*stripPhiPitch - order*stripPhiPitch/4.;
  if ( processor==6 || phiHalfStrip0<0) phiHalfStrip0 += 2*M_PI;
  int offsetLoc = lround( (phiHalfStrip0-phi15deg)/hsPhiPitch );

  int halfStrip = digi.getStrip(); // returns halfStrip 0..159
  //FIXME: to be checked (only important for ME1/3) keep more bits for offset, truncate at the end
  int phi = offsetLoc + order*scale*halfStrip;

  return phi;
}

///////////////////////////////////////
///////////////////////////////////////
int AngleConverter::getProcessorPhi(unsigned int iProcessor, l1t::tftype part, const RPCDetId & rollId, const unsigned int &digi) const
{

  const double hsPhiPitch = 2*M_PI/nPhiBins;
  const int dummy = nPhiBins;
  int processor = iProcessor+1;
  const RPCRoll* roll = _georpc->roll(rollId);
  if (!roll) return dummy;

  double phi15deg =  M_PI/3.*(processor-1)+M_PI/12.;                    // "0" is 15degree moved cyclicaly to each processor, note [0,2pi]
  double stripPhi = (roll->toGlobal(roll->centreOfStrip((int)digi))).phi(); // note [-pi,pi]

  // adjust [0,2pi] and [-pi,pi] to get deltaPhi difference properly
  switch (processor) {
    case 1: break;
    case 6: {phi15deg -= 2*M_PI; break; }
    default : {if (stripPhi < 0) stripPhi += 2*M_PI; break; }
  }

  // local angle in CSC halfStrip usnits
  int halfStrip = lround ( (stripPhi-phi15deg)/hsPhiPitch );
    
  return halfStrip;
}
///////////////////////////////////////
///////////////////////////////////////
int AngleConverter::getGlobalEta(unsigned int rawid,
				 const L1MuDTChambPhDigi &aDigi,
				 const L1MuDTChambThContainer *dtThDigis){

  
  const DTChamberId baseid(aDigi.whNum(),aDigi.stNum(),aDigi.scNum()+1);
  
  // do not use this pointer for anything other than creating a trig geom
  std::unique_ptr<DTChamber> chamb(const_cast<DTChamber*>(_geodt->chamber(baseid)));
  
  std::unique_ptr<DTTrigGeom> trig_geom( new DTTrigGeom(chamb.get(),false) );
  chamb.release(); // release it here so no one gets funny ideas  
  // super layer one is the theta superlayer in a DT chamber
  // station 4 does not have a theta super layer
  // the BTI index from the theta trigger is an OR of some BTI outputs
  // so, we choose the BTI that's in the middle of the group
  // as the BTI that we get theta from
  // TODO:::::>>> need to make sure this ordering doesn't flip under wheel sign
  const int NBTI_theta = ( (baseid.station() != 4) ? 
			   trig_geom->nCell(2) : trig_geom->nCell(3) );
  const int bti_group = findBTIgroup(aDigi,dtThDigis);
  const unsigned bti_actual = bti_group*NBTI_theta/7 + NBTI_theta/14 + 1;  
  DTBtiId thetaBTI;  
  if ( baseid.station() != 4 && bti_group != -1) {
    thetaBTI = DTBtiId(baseid,2,bti_actual);
  } else {
    // since this is phi oriented it'll give us theta in the middle
    // of the chamber
    thetaBTI = DTBtiId(baseid,3,1); 
  }
  const GlobalPoint theta_gp = trig_geom->CMSPosition(thetaBTI);
  return etaVal2Code(theta_gp.eta());
//  int iEta = theta_gp.eta()/2.61*240;
//  return iEta;
}
///////////////////////////////////////
///////////////////////////////////////
int AngleConverter::getGlobalEta(unsigned int rawid, const CSCCorrelatedLCTDigi &aDigi){

   ///Code taken from GeometryTranslator.
  ///Will be replaced by direct CSC phi local to global scale
  ///transformation as used in FPGA implementation

  
  // alot of this is transcription and consolidation of the CSC
  // global phi calculation code
  // this works directly with the geometry 
  // rather than using the old phi luts
  const CSCDetId id(rawid); 
  // we should change this to weak_ptrs at some point
  // requires introducing std::shared_ptrs to geometry
  std::unique_ptr<const CSCChamber> chamb(_geocsc->chamber(id));
  std::unique_ptr<const CSCLayerGeometry> layer_geom(
						     chamb->layer(CSCConstants::KEY_ALCT_LAYER)->geometry()
						     );
  std::unique_ptr<const CSCLayer> layer(
					chamb->layer(CSCConstants::KEY_ALCT_LAYER)
					);
  
  const uint16_t halfstrip = aDigi.getStrip();
  const uint16_t pattern = aDigi.getPattern();
  const uint16_t keyWG = aDigi.getKeyWG();
  //const unsigned maxStrips = layer_geom->numberOfStrips();  

  // so we can extend this later 
  // assume TMB2007 half-strips only as baseline
  double offset = 0.0;
  switch(1) {
  case 1:
    offset = CSCPatternLUT::get2007Position(pattern);
  }
  const unsigned halfstrip_offs = unsigned(0.5 + halfstrip + offset);
  const unsigned strip = halfstrip_offs/2 + 1; // geom starts from 1

  // the rough location of the hit at the ALCT key layer
  // we will refine this using the half strip information
  const LocalPoint coarse_lp = 
    layer_geom->stripWireGroupIntersection(strip,keyWG);  
  const GlobalPoint coarse_gp = layer->surface().toGlobal(coarse_lp);  
  
  // the strip width/4.0 gives the offset of the half-strip
  // center with respect to the strip center
  const double hs_offset = layer_geom->stripPhiPitch()/4.0;
  
  // determine handedness of the chamber
  const bool ccw = isCSCCounterClockwise(layer);
  // we need to subtract the offset of even half strips and add the odd ones
  const double phi_offset = ( ( halfstrip_offs%2 ? 1 : -1)*
			      ( ccw ? -hs_offset : hs_offset ) );
  
  // the global eta calculation uses the middle of the strip
  // so no need to increment it
  const GlobalPoint final_gp( GlobalPoint::Polar( coarse_gp.theta(),
						  (coarse_gp.phi().value() + 
						   phi_offset),
						  coarse_gp.mag() ) );  
  // release ownership of the pointers
  chamb.release();
  layer_geom.release();
  layer.release();

//  std::cout <<id<<" st: " << id.station()<< "ri: "<<id.ring()<<" eta: " <<  final_gp.eta() 
//           <<" etaCode_simple: " <<  etaVal2Code( final_gp.eta() )<< " KW: "<<keyWG <<" etaKeyWG2Code: "<<etaKeyWG2Code(id,keyWG)<< std::endl;

   return etaKeyWG2Code(id,keyWG);

// return etaVal2Code( final_gp.eta() );
// int iEta =  final_gp.eta()/2.61*240;
// return iEta;  
}
///////////////////////////////////////
///////////////////////////////////////
int AngleConverter::getGlobalEta(unsigned int rawid, const unsigned int &strip){

  const RPCDetId id(rawid);
  std::unique_ptr<const RPCRoll>  roll(_georpc->roll(id));
  const LocalPoint lp = roll->centreOfStrip((int)strip);
  const GlobalPoint gp = roll->toGlobal(lp);
  roll.release();

  return etaVal2Code(gp.eta());
//  float iEta = gp.eta()/2.61*240;
//  return iEta;

} 
///////////////////////////////////////
///////////////////////////////////////
bool AngleConverter::
isCSCCounterClockwise(const std::unique_ptr<const CSCLayer>& layer) const {
  const int nStrips = layer->geometry()->numberOfStrips();
  const double phi1 = layer->centerOfStrip(1).phi();
  const double phiN = layer->centerOfStrip(nStrips).phi();
  return ( (std::abs(phi1 - phiN) < M_PI  && phi1 >= phiN) || 
	   (std::abs(phi1 - phiN) >= M_PI && phi1 < phiN)     );  
}
///////////////////////////////////////
///////////////////////////////////////
const int AngleConverter::findBTIgroup(const L1MuDTChambPhDigi &aDigi,
				       const L1MuDTChambThContainer *dtThDigis){

  int bti_group = -1;
  
  const L1MuDTChambThDigi *theta_segm = dtThDigis->chThetaSegm(aDigi.whNum(),
							       aDigi.stNum(),
							       aDigi.scNum(),
							       aDigi.bxNum());
  if(!theta_segm) return  bti_group;
  
  for(unsigned int i = 0; i < 7; ++i ){
    if(theta_segm->position(i) && bti_group<0) bti_group = i;
    ///If there are more than one theta digi we do not take is
    ///due to unresolvet ambiguity. In this case we take eta of the
    ///middle of the chamber.
    else if(theta_segm->position(i) && bti_group>-1) return -1;
  }
      
  return bti_group;
}
///////////////////////////////////////
///////////////////////////////////////
