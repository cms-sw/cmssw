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

AngleConverter::AngleConverter(): _geom_cache_id(0ULL) { }
///////////////////////////////////////
///////////////////////////////////////
AngleConverter::~AngleConverter() {  }
///////////////////////////////////////
///////////////////////////////////////
void AngleConverter::checkAndUpdateGeometry(const edm::EventSetup& es) {
  const MuonGeometryRecord& geom = es.get<MuonGeometryRecord>();
  unsigned long long geomid = geom.cacheIdentifier();
  if( _geom_cache_id != geomid ) {
    geom.get(_georpc);  
    geom.get(_geocsc);    
    geom.get(_geodt);
    _geom_cache_id = geomid;
  }  
}
///////////////////////////////////////
///////////////////////////////////////
float AngleConverter::getGlobalPhi(unsigned int rawid, const RPCDigi & aDigi){

  ///Will be replaced by LUT based
  ///transformation as used in FPGA implementation

  const RPCDetId id(rawid);
  std::unique_ptr<const RPCRoll>  roll(_georpc->roll(id));
  const uint16_t strip = aDigi.strip();
  const LocalPoint lp = roll->centreOfStrip(strip);
  const GlobalPoint gp = roll->toGlobal(lp);
  roll.release();

  //float phi = gp.phi()/(2.0*M_PI);
  //int iPhi = phi*OMTFConfiguration::nPhiBins;

  return gp.phi();
}
///////////////////////////////////////
///////////////////////////////////////
int AngleConverter::getGlobalPhi(unsigned int rawid, const L1MuDTChambPhDigi & aDigi){

  // local phi in sector -> global phi
  float phi = aDigi.phi()/4096.0; 
  phi += aDigi.scNum()*M_PI/6.0; // add sector offset
  if(phi>M_PI) phi-=2*M_PI;
  phi/=2.0*M_PI;
      
  int iPhi = phi*OMTFConfiguration::nPhiBins;
   
  return iPhi;
}
///////////////////////////////////////
///////////////////////////////////////
int AngleConverter::getGlobalPhi(unsigned int rawid, const CSCCorrelatedLCTDigi & aDigi){

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

  float phi = final_gp.phi()/(2.0*M_PI);
  int iPhi =  phi*OMTFConfiguration::nPhiBins;
  
  return iPhi;
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
  int iEta = theta_gp.eta()/2.61*240;
  return iEta;
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

  int iEta =  final_gp.eta()/2.61*240;
  return iEta;  
}
///////////////////////////////////////
///////////////////////////////////////
int AngleConverter::getGlobalEta(unsigned int rawid, const RPCDigi &aDigi){

  const RPCDetId id(rawid);
  std::unique_ptr<const RPCRoll>  roll(_georpc->roll(id));
  const uint16_t strip = aDigi.strip();
  const LocalPoint lp = roll->centreOfStrip(strip);
  const GlobalPoint gp = roll->toGlobal(lp);
  roll.release();

  float iEta = gp.eta()/2.61*240;

  return iEta;

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
