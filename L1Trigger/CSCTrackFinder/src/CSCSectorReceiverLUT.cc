#include <L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverMiniLUT.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeomManager.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCPatternLUT.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCFrontRearLUT.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCBitWidths.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCTFConstants.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>

#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <fstream>
#include <cstring>

lclphidat* CSCSectorReceiverLUT::me_lcl_phi = NULL;
bool CSCSectorReceiverLUT::me_lcl_phi_loaded = false;


CSCSectorReceiverLUT::CSCSectorReceiverLUT(int endcap, int sector, int subsector, int station,
					   const edm::ParameterSet & pset, bool TMB07):_endcap(endcap),_sector(sector),
									   _subsector(subsector),
									   _station(station),isTMB07(TMB07)
{
  LUTsFromFile = pset.getUntrackedParameter<bool>("ReadLUTs",false);
  useMiniLUTs = pset.getUntrackedParameter<bool>("UseMiniLUTs", true);
  isBinary = pset.getUntrackedParameter<bool>("Binary",false);

  me_global_eta = NULL;
  me_global_phi = NULL;
  mb_global_phi = NULL;
  if(LUTsFromFile && !useMiniLUTs)
    {
      me_lcl_phi_file = pset.getUntrackedParameter<edm::FileInPath>("LocalPhiLUT", edm::FileInPath(std::string("L1Trigger/CSCTrackFinder/LUTs/LocalPhiLUT"
													       + (isBinary ? std::string(".bin") : std::string(".dat")))));
      me_gbl_phi_file = pset.getUntrackedParameter<edm::FileInPath>("GlobalPhiLUTME", edm::FileInPath((std::string("L1Trigger/CSCTrackFinder/LUTs/GlobalPhiME")
												       + encodeFileIndex()
												       + (isBinary ? std::string(".bin") : std::string(".dat")))));
      if(station == 1)
	mb_gbl_phi_file = pset.getUntrackedParameter<edm::FileInPath>("GlobalPhiLUTMB", edm::FileInPath((std::string("L1Trigger/CSCTrackFinder/LUTs/GlobalPhiMB")
													 + encodeFileIndex()
													 + (isBinary ? std::string(".bin") : std::string(".dat")))));
      me_gbl_eta_file = pset.getUntrackedParameter<edm::FileInPath>("GlobalEtaLUTME", edm::FileInPath((std::string("L1Trigger/CSCTrackFinder/LUTs/GlobalEtaME")
												       + encodeFileIndex()
												       + (isBinary ? std::string(".bin") : std::string(".dat")))));
      readLUTsFromFile();
    }

}

CSCSectorReceiverLUT::CSCSectorReceiverLUT(const CSCSectorReceiverLUT& lut):_endcap(lut._endcap),
									    _sector(lut._sector),
									    _subsector(lut._subsector),
									    _station(lut._station),
									    me_lcl_phi_file(lut.me_lcl_phi_file),
									    me_gbl_phi_file(lut.me_gbl_phi_file),
									    mb_gbl_phi_file(lut.mb_gbl_phi_file),
									    me_gbl_eta_file(lut.me_gbl_eta_file),
									    LUTsFromFile(lut.LUTsFromFile),
									    isBinary(lut.isBinary)
{
  if(lut.mb_global_phi)
    {
      mb_global_phi = new gblphidat[1<<CSCBitWidths::kGlobalPhiAddressWidth];
      memcpy(mb_global_phi, lut.mb_global_phi, (1<<CSCBitWidths::kGlobalPhiAddressWidth)*sizeof(gblphidat));
    }
  else mb_global_phi = NULL;
  if(lut.me_global_phi)
    {
      me_global_phi = new gblphidat[1<<CSCBitWidths::kGlobalPhiAddressWidth];
      memcpy(me_global_phi, lut.me_global_phi, (1<<CSCBitWidths::kGlobalPhiAddressWidth)*sizeof(gblphidat));
    }
  else me_global_phi = NULL;
  if(lut.me_global_eta)
    {
      me_global_eta = new gbletadat[1<<CSCBitWidths::kGlobalEtaAddressWidth];
      memcpy(me_global_eta, lut.me_global_eta, (1<<CSCBitWidths::kGlobalEtaAddressWidth)*sizeof(gbletadat));
    }
  else me_global_eta = NULL;
}

CSCSectorReceiverLUT& CSCSectorReceiverLUT::operator=(const CSCSectorReceiverLUT& lut)
{
  if(this != &lut)
    {
      _endcap = lut._endcap;
      _sector = lut._sector;
      _subsector = lut._subsector;
      _station = lut._station;
      me_lcl_phi_file = lut.me_lcl_phi_file;
      me_gbl_phi_file = lut.me_gbl_phi_file;
      mb_gbl_phi_file = lut.mb_gbl_phi_file;
      me_gbl_eta_file = lut.me_gbl_eta_file;
      LUTsFromFile = lut.LUTsFromFile;
      isBinary = lut.isBinary;

      if(lut.mb_global_phi)
	{
	  mb_global_phi = new gblphidat[1<<CSCBitWidths::kGlobalPhiAddressWidth];
	  memcpy(mb_global_phi, lut.mb_global_phi, (1<<CSCBitWidths::kGlobalPhiAddressWidth)*sizeof(gblphidat));
	}
      else mb_global_phi = NULL;

      if(lut.me_global_phi)
	{
	  me_global_phi = new gblphidat[1<<CSCBitWidths::kGlobalPhiAddressWidth];
	  memcpy(me_global_phi, lut.me_global_phi, (1<<CSCBitWidths::kGlobalPhiAddressWidth)*sizeof(gblphidat));
	}
      else me_global_phi = NULL;

      if(lut.me_global_eta)
	{
	  me_global_eta = new gbletadat[1<<CSCBitWidths::kGlobalEtaAddressWidth];
	  memcpy(me_global_eta, lut.me_global_eta, (1<<CSCBitWidths::kGlobalEtaAddressWidth)*sizeof(gbletadat));
	}
      else me_global_eta = NULL;
    }
  return *this;
}

CSCSectorReceiverLUT::~CSCSectorReceiverLUT()
{
  if(me_lcl_phi_loaded)
    {
      delete me_lcl_phi;
      me_lcl_phi = NULL;
      me_lcl_phi_loaded = false;
    }
  if(me_global_eta)
    {
      delete me_global_eta;
      me_global_eta = NULL;
    }
  if(me_global_phi)
    {
      delete me_global_phi;
      me_global_phi = NULL;
    }
  if(mb_global_phi)
    {
      delete mb_global_phi;
      mb_global_phi = NULL;
    }
}

lclphidat CSCSectorReceiverLUT::calcLocalPhi(const lclphiadd& theadd) const
{
  lclphidat data;

  constexpr int maxPhiL = 1<<CSCBitWidths::kLocalPhiDataBitWidth;
  double binPhiL = static_cast<double>(maxPhiL)/(2.*CSCConstants::MAX_NUM_STRIPS);

  memset(&data,0,sizeof(lclphidat));

  double patternOffset;

  if(isTMB07) patternOffset = CSCPatternLUT::get2007Position((theadd.pattern_type<<3) + theadd.clct_pattern);
  else patternOffset = CSCPatternLUT::getPosition(theadd.clct_pattern);

  // The phiL value stored is for the center of the half-/di-strip.
  if(theadd.strip < 2*CSCConstants::MAX_NUM_STRIPS)
    if(theadd.pattern_type == 1 || isTMB07) // if halfstrip (Note: no distrips in TMB 2007 patterns)
      data.phi_local = static_cast<unsigned>((0.5 + theadd.strip + patternOffset)*binPhiL);
    else // if distrip
      data.phi_local = static_cast<unsigned>((2 + theadd.strip + 4.*patternOffset)*binPhiL);
  else {
    throw cms::Exception("CSCSectorReceiverLUT")
      << "+++ Value of strip, " << theadd.strip
      << ", exceeds max allowed, " << 2*CSCConstants::MAX_NUM_STRIPS-1
      << " +++\n";
  }

  if (data.phi_local >= maxPhiL) {
    throw cms::Exception("CSCSectorReceiverLUT")
      << "+++ Value of phi_local, " << data.phi_local
      << ", exceeds max allowed, " << maxPhiL-1 << " +++\n";
  }

  LogDebug("CSCSectorReceiver")
    << "endcap = " << _endcap << " station = " << _station
    << " maxPhiL = " << maxPhiL << " binPhiL = " << binPhiL;
  LogDebug("CSCSectorReceiver")
    << "strip # " << theadd.strip << " hs/ds = " << theadd.pattern_type
    << " pattern = " << theadd.clct_pattern << " offset = " << patternOffset
    << " phi_local = " << data.phi_local;

  /// Local Phi Bend is always zero. Until we start using it.
  data.phi_bend_local = 0;

  return data; //return LUT result
}


void CSCSectorReceiverLUT::fillLocalPhiLUT()
{
  // read data in from a file... Add this later.
}

lclphidat CSCSectorReceiverLUT::localPhi(const int strip, const int pattern,
					 const int quality, const int lr) const
{
  lclphiadd theadd;

  theadd.strip = strip;
  theadd.clct_pattern = pattern & 0x7;
  theadd.pattern_type = (pattern & 0x8) >> 3;
  theadd.quality = quality;
  theadd.lr = lr;
  theadd.spare = 0;

  return localPhi(theadd);
}

lclphidat CSCSectorReceiverLUT::localPhi(unsigned address) const
{
  lclphidat result;
  lclphiadd theadd(address);
  
  if(useMiniLUTs && isTMB07)
    {
      result = CSCSectorReceiverMiniLUT::calcLocalPhiMini(address);
    }
  else if(LUTsFromFile) result = me_lcl_phi[address];
  else result = calcLocalPhi(theadd);

  return result;
}

lclphidat CSCSectorReceiverLUT::localPhi(lclphiadd address) const
{
  lclphidat result;

  if(useMiniLUTs && isTMB07)
    {
      result = CSCSectorReceiverMiniLUT::calcLocalPhiMini(address.toint()); 
    }
  else if(LUTsFromFile) result = me_lcl_phi[address.toint()];
  else result = calcLocalPhi(address);

  return result;
}

double CSCSectorReceiverLUT::getGlobalPhiValue(const CSCLayer* thelayer, const unsigned& strip, const unsigned& wire_group) const
{
  double result = 0.0;
  //CSCLayerGeometry* thegeom;
  //LocalPoint lp;
  //GlobalPoint gp;

  try
    {
      //thegeom = const_cast<CSCLayerGeometry*>(thelayer->geometry());
      //lp = thegeom->stripWireGroupIntersection(strip, wire_group);
      //gp = thelayer->surface().toGlobal(lp);
      result = thelayer->centerOfStrip(strip).phi();//gp.phi();

      if (result < 0.) result += 2.*M_PI;
    }
  catch(edm::Exception& e)
    {
      LogDebug("CSCSectorReceiverLUT|getGlobalPhiValue") << e.what();
    }

  return result;
}

gblphidat CSCSectorReceiverLUT::calcGlobalPhiME(const gblphiadd& address) const
{
  gblphidat result(0);
  CSCTriggerGeomManager* thegeom = CSCTriggerGeometry::get();
  CSCChamber* thechamber = NULL;
  const CSCLayer* thelayer = NULL;
  const CSCLayerGeometry* layergeom = NULL;
  int cscid = address.cscid;
  unsigned wire_group = address.wire_group;
  unsigned local_phi = address.phi_local;
  const double sectorOffset = (CSCTFConstants::SECTOR1_CENT_RAD-CSCTFConstants::SECTOR_RAD/2.) + (_sector-1)*M_PI/3.;

  //Number of global phi units per radian.
  constexpr int maxPhiG = 1<<CSCBitWidths::kGlobalPhiDataBitWidth;
  double binPhiG = static_cast<double>(maxPhiG)/CSCTFConstants::SECTOR_RAD;

  // We will use these to convert the local phi into radians.
  constexpr unsigned int maxPhiL = 1<<CSCBitWidths::kLocalPhiDataBitWidth;
  const double binPhiL = static_cast<double>(maxPhiL)/(2.*CSCConstants::MAX_NUM_STRIPS);

  if(cscid < CSCTriggerNumbering::minTriggerCscId())
    {
      edm::LogWarning("CSCSectorReceiverLUT|getGlobalPhiValue")
	<< " warning: cscId " << cscid << " is out of bounds ["
	<< CSCTriggerNumbering::maxTriggerCscId() << "-"
	<< CSCTriggerNumbering::maxTriggerCscId() << "]\n";
      throw cms::Exception("CSCSectorReceiverLUT")
	<< "+++ Value of CSC ID, " << cscid
	<< ", is out of bounds [" << CSCTriggerNumbering::minTriggerCscId() << "-"
	<< CSCTriggerNumbering::maxTriggerCscId() << "] +++\n";
    }

  if(cscid > CSCTriggerNumbering::maxTriggerCscId())
    {
      edm::LogWarning("CSCSectorReceiverLUT|getGlobalPhiValue")
	<< " warning: cscId " << cscid << " is out of bounds ["
	<< CSCTriggerNumbering::maxTriggerCscId() << "-"
	<< CSCTriggerNumbering::maxTriggerCscId() << "]\n";
      throw cms::Exception("CSCSectorReceiverLUT")
	<< "+++ Value of CSC ID, " << cscid
	<< ", is out of bounds [" << CSCTriggerNumbering::minTriggerCscId() << "-"
	<< CSCTriggerNumbering::maxTriggerCscId() << "] +++\n";
    }

  if(wire_group >= 1<<5)
    {
      edm::LogWarning("CSCSectorReceiverLUT|getGlobalPhiValue")
	<< "warning: wire_group" << wire_group
	<< " is out of bounds (1-" << ((1<<5)-1) << "]\n";
      throw cms::Exception("CSCSectorReceiverLUT")
	<< "+++ Value of wire_group, " << wire_group
	<< ", is out of bounds (1-" << ((1<<5)-1) << "] +++\n";
    }

  if(local_phi >= maxPhiL)
    {
      edm::LogWarning("CSCSectorReceiverLUT|getGlobalPhiValue")
	<< "warning: local_phi" << local_phi
	<< " is out of bounds [0-" << maxPhiL << ")\n";
      throw cms::Exception("CSCSectorReceiverLUT")
	<< "+++ Value of local_phi, " << local_phi
	<< ", is out of bounds [0-, " << maxPhiL << ") +++\n";
    }

  try
    {
      thechamber = thegeom->chamber(_endcap,_station,_sector,_subsector,cscid);
      if(thechamber)
	{
	  if(isTMB07)
	    {
	      layergeom = thechamber->layer(CSCConstants::KEY_CLCT_LAYER)->geometry();
	      thelayer = thechamber->layer(CSCConstants::KEY_CLCT_LAYER);
	    }
	  else
	    {
	      layergeom = thechamber->layer(CSCConstants::KEY_CLCT_LAYER_PRE_TMB07)->geometry();
	      thelayer = thechamber->layer(CSCConstants::KEY_CLCT_LAYER_PRE_TMB07);
	    }
	  const int nStrips = layergeom->numberOfStrips();
	  // PhiL is the strip number converted into some units between 0 and
	  // 1023.  When we did the conversion in fillLocalPhiTable(), we did
	  // not know for which chamber we do it (and, therefore, how many strips
	  // it has), and always used the maximum possible number of strips
	  // per chamber, MAX_NUM_STRIPS=80.  Now, since we know the chamber id
	  // and how many strips the chamber has, we can re-adjust the scale.
	  //const double scale = static_cast<double>(CSCConstants::MAX_NUM_STRIPS)/nStrips;

	  int strip = 0, halfstrip = 0;

          halfstrip = static_cast<int>(local_phi/binPhiL);
          strip     = halfstrip/2;

	  // Find the phi width of the chamber and the position of its "left"
	  // (lower phi) edge (both in radians).
	  // Phi positions of the centers of the first and of the last strips
	  // in the chamber.
	  const double phi_f = getGlobalPhiValue(thelayer, 1, wire_group);
	  const double phi_l = getGlobalPhiValue(thelayer, nStrips, wire_group);
	  // Phi widths of the half-strips at both ends of the chamber;
	  // surprisingly, they are not the same.
	  const double hsWidth_f = fabs(getGlobalPhiValue(thelayer, 2, wire_group) - phi_f)/2.;
	  const double hsWidth_l = fabs(phi_l - getGlobalPhiValue(thelayer, nStrips - 1, wire_group))/2.;

	  // The "natural" match between the strips and phi values -- when
	  // a larger strip number corresponds to a larger phi value, i.e. strips
	  // are counted clockwise if we look at them from the inside of the
	  // detector -- is reversed for some stations.  At the moment, these
	  // are stations 3 and 4 of the 1st endcap, and stations 1 and 2 of
	  // the 2nd endcap.  Instead of using
	  // if ((theEndcap == 1 && theStation <= 2) ||
	  // (theEndcap == 2 && theStation >= 3)),
	  // we get the order from the phi values of the first and the last strip
	  // in a chamber, just in case the counting scheme changes in the future.
	  // Once we know how the strips are counted, we can go from the middle
	  // of the strips to their outer edges.
	  bool   clockwiseOrder;
	  double leftEdge, rightEdge;
	  if (fabs(phi_f - phi_l) < M_PI)
	    {
	      if (phi_f < phi_l) clockwiseOrder = true;
	      else clockwiseOrder = false;
	    }
	  else
	    { // the chamber crosses the phi = pi boundary
	      if (phi_f < phi_l) clockwiseOrder = false;
	      else clockwiseOrder = true;
	    }
	  if (clockwiseOrder)
	    {
	      leftEdge  = phi_f - hsWidth_f;
	      rightEdge = phi_l + hsWidth_l;
	    }
	  else
	    {
	      leftEdge  = phi_l - hsWidth_l;
	      rightEdge = phi_f + hsWidth_f;
	    }
	  if (fabs(phi_f - phi_l) >= M_PI) {rightEdge += 2.*M_PI;}
	  //double chamberWidth = (rightEdge - leftEdge);

	  // Chamber offset, relative to the edge of the sector.
	  //double chamberOffset = leftEdge - sectorOffset;
	  //if (chamberOffset < -M_PI) chamberOffset += 2*M_PI;

	  double temp_phi = 0.0, strip_phi = 0.0, delta_phi = 0.0;
	  double distFromHalfStripCenter = 0.0, halfstripWidth = 0.0;

	  if (strip < nStrips)
	    {
	      // Approximate distance from the center of the half-strip to the center
	      // of this phil bin, in units of half-strip width.
	      distFromHalfStripCenter = (local_phi+0.5)/binPhiL - halfstrip - 0.5;
	      // Half-strip width (in rad), calculated as the half-distance between
	      // the adjacent strips.  Since in the current ORCA implementation
	      // the half-strip width changes from strip to strip, base the choice
	      // of the adjacent strip on the half-strip number.
	      if ((halfstrip%2 == 0 && halfstrip != 0) || halfstrip == 2*nStrips-1) {
		halfstripWidth =
		  fabs(getGlobalPhiValue(thelayer, strip+1, wire_group) - getGlobalPhiValue(thelayer, strip, wire_group)) / 2.;
	      }
	      else
		{
		  halfstripWidth =
		    fabs(getGlobalPhiValue(thelayer, strip+1, wire_group) - getGlobalPhiValue(thelayer, strip+2, wire_group)) / 2.;
		}
	      // Correction for the strips crossing the 180 degree boundary.
	      if (halfstripWidth > M_PI/2.) halfstripWidth = M_PI - halfstripWidth;
	      // Phi at the center of the strip.
	      strip_phi = getGlobalPhiValue(thelayer, strip+1, wire_group);
	      // Distance between the center of the strip and the phil position.
	      delta_phi = halfstripWidth*(((halfstrip%2)-0.5)+distFromHalfStripCenter);
	      if (clockwiseOrder)
		temp_phi = strip_phi+ delta_phi;
	      else
		temp_phi = strip_phi- delta_phi;
	    }
	  else
	    {
	      // PhiL values that do not have corresponding strips (the chamber
	      // has less than 80 strips assumed in fillLocalPhi).  It does not
	      // really matter what we do with these values; at the moment, just
	      // set them to the phis of the edges of the chamber.
	      if (clockwiseOrder) temp_phi = rightEdge;
	      else temp_phi = leftEdge;
	    }

	  // Finally, subtract the sector offset and convert to the scale of
	  // the global phi.

	  temp_phi -= sectorOffset;

	  if (temp_phi < 0.) temp_phi += 2.*M_PI;

	  temp_phi *= binPhiG;

	  if (temp_phi < 0.)
	    {
	      result.global_phi = 0;
	    }
	  else if (temp_phi >= maxPhiG)
	    {
	      result.global_phi = maxPhiG - 1;
	    }
	  else
	    {
	     result.global_phi = static_cast<unsigned short>(temp_phi);
	    }

	  LogDebug("CSCSectorReceiverLUT")
	    << "local_phi = " << local_phi
	    << " halfstrip = " << halfstrip << " strip = " << strip
	    << " distFromHalfStripCenter = " << distFromHalfStripCenter
	    << " halfstripWidth = " << halfstripWidth
	    << " strip phi = " << strip_phi/(M_PI/180.)
	    << " temp_phi = " << temp_phi*CSCTFConstants::SECTOR_DEG/maxPhiG
	    << " global_phi = "    << result.global_phi
	    << " " << result.global_phi*CSCTFConstants::SECTOR_DEG/maxPhiG;

	}
    }
  catch(edm::Exception& e)
    {
      edm::LogError("CSCSectorReceiverLUT|getGlobalPhiValue") << e.what();
    }

  return result;
}

gblphidat CSCSectorReceiverLUT::globalPhiME(int phi_local, int wire_group, int cscid) const
{
  gblphidat result;
  gblphiadd theadd;
  theadd.phi_local = phi_local;
  theadd.wire_group = ((1<<5)-1)&(wire_group >> 2); // want 2-7 of wg
  theadd.cscid = cscid;

  if(useMiniLUTs && isTMB07) result = CSCSectorReceiverMiniLUT::calcGlobalPhiMEMini(_endcap, _sector, _station, _subsector, theadd.toint());
  else if(LUTsFromFile) result = me_global_phi[theadd.toint()];
  else result = calcGlobalPhiME(theadd);

  return result;
}

gblphidat CSCSectorReceiverLUT::globalPhiME(unsigned address) const
{
  gblphidat result;

  if(useMiniLUTs && isTMB07) result = CSCSectorReceiverMiniLUT::calcGlobalPhiMEMini(_endcap, _sector, _station, _subsector, address);
  else if(LUTsFromFile) result = me_global_phi[address];
  else result = calcGlobalPhiME(gblphiadd(address));

  return result;
}

gblphidat CSCSectorReceiverLUT::globalPhiME(gblphiadd address) const
{
  gblphidat result;

  if(useMiniLUTs && isTMB07) result = CSCSectorReceiverMiniLUT::calcGlobalPhiMEMini(_endcap, _sector, _station, _subsector, address.toint());
  else if(LUTsFromFile) result = me_global_phi[address.toint()];
  else result = calcGlobalPhiME(address);

  return result;
}

gblphidat CSCSectorReceiverLUT::calcGlobalPhiMB(const gblphidat &csclut) const
{
  gblphidat dtlut;

  // The following method was ripped from D. Holmes' LUT conversion program
  // modifications from Darin and GP
  int GlobalPhiMin = (_subsector == 1) ? 0x42 : 0x800;  // (0.999023 : 31 in degrees)
  int GlobalPhiMax = (_subsector == 1) ? 0x7ff : 0xfbd; // (30.985 : 60.986 in degrees)
  double GlobalPhiShift = (1.0*GlobalPhiMin + (GlobalPhiMax - GlobalPhiMin)/2.0);

  double dt_out = static_cast<double>(csclut.global_phi) - GlobalPhiShift;

  // these numbers are 62 deg / 1 rad (CSC phi scale vs. DT phi scale)
  dt_out = (dt_out/1982)*2145; //CSC phi 62 degrees; DT phi 57.3 degrees

  if(dt_out >= 0) // msb != 1
    {
      dtlut.global_phi = 0x7ff&static_cast<unsigned>(dt_out);
    }
  else
    {
      dtlut.global_phi = static_cast<unsigned>(-dt_out);
      dtlut.global_phi = ~dtlut.global_phi;
      dtlut.global_phi |= 0x800;
    }

  return dtlut;
}

gblphidat CSCSectorReceiverLUT::globalPhiMB(int phi_local,int wire_group, int cscid) const
{
  gblphiadd address;
  gblphidat result;

  address.cscid = cscid;
  address.wire_group = ((1<<5)-1)&(wire_group>>2);
  address.phi_local = phi_local;

  // comment for now
  //  if(useMiniLUTs && isTMB07) result = CSCSectorReceiverMiniLUT::calcGlobalPhiMBMini(_endcap, _sector, _subsector, address.toint());
  //else 
  if(LUTsFromFile) result = mb_global_phi[address.toint()];
  else result = calcGlobalPhiMB(globalPhiME(address));

  return result;
}

gblphidat CSCSectorReceiverLUT::globalPhiMB(unsigned address) const
{
  gblphidat result;
  gblphiadd theadd(address);

  //if(useMiniLUTs && isTMB07) result = CSCSectorReceiverMiniLUT::calcGlobalPhiMBMini(_endcap, _sector, _subsector, address);
  //else 
  if(LUTsFromFile) result = mb_global_phi[theadd.toint()];
  else result = calcGlobalPhiMB(globalPhiME(address));

  return result;
}

gblphidat CSCSectorReceiverLUT::globalPhiMB(gblphiadd address) const
{
  gblphidat result;

  //if(useMiniLUTs && isTMB07) result = CSCSectorReceiverMiniLUT::calcGlobalPhiMBMini(_endcap, _sector, _subsector, address.toint());
  //else 
  if(LUTsFromFile) result = mb_global_phi[address.toint()];
  else result = calcGlobalPhiMB(globalPhiME(address));

  return result;
}

double CSCSectorReceiverLUT::getGlobalEtaValue(const unsigned& thecscid, const unsigned& thewire_group, const unsigned& thephi_local) const
{
  double result = 0.0;
  unsigned wire_group = thewire_group;
  int cscid = thecscid;
  unsigned phi_local = thephi_local;

  // Flag to be set if one wants to apply phi corrections ONLY in ME1/1.
  // Turn it into a parameter?
  bool me1ir_only = false;

  if(cscid < CSCTriggerNumbering::minTriggerCscId() ||
     cscid > CSCTriggerNumbering::maxTriggerCscId()) {
       edm::LogWarning("CSCSectorReceiverLUT|getEtaValue")
	 << " warning: cscId " << cscid
	 << " is out of bounds [1-" << CSCTriggerNumbering::maxTriggerCscId()
	 << "]\n";
      cscid = CSCTriggerNumbering::maxTriggerCscId();
    }

  CSCTriggerGeomManager* thegeom = CSCTriggerGeometry::get();
  CSCLayerGeometry* layerGeom = NULL;
  const unsigned numBins = 1 << 2; // 4 local phi bins

  if(phi_local > numBins - 1) {
      edm::LogWarning("CSCSectorReceiverLUT|getEtaValue")
	<< "warning: phiL " << phi_local
	<< " is out of bounds [0-" << numBins - 1 << "]\n";
      phi_local = numBins - 1;
  }
  try
    {
      const CSCChamber* thechamber = thegeom->chamber(_endcap,_station,_sector,_subsector,cscid);
      if(thechamber) {
	layerGeom = const_cast<CSCLayerGeometry*>(thechamber->layer(CSCConstants::KEY_ALCT_LAYER)->geometry());
	const unsigned nWireGroups = layerGeom->numberOfWireGroups();

	// Check wire group numbers; expect them to be counted from 0, as in
	// CorrelatedLCTDigi class.
	if (wire_group >= nWireGroups) {
	   edm::LogWarning("CSCSectorReceiverLUT|getEtaValue")
	     << "warning: wireGroup " << wire_group
	    << " is out of bounds [0-" << nWireGroups << ")\n";
	  wire_group = nWireGroups - 1;
	}
	// Convert to [1; nWireGroups] range used in geometry methods.
	wire_group += 1;

	// If me1ir_only is set, apply phi corrections only in ME1/1.
	if (me1ir_only &&
	    (_station != 1 ||
	     CSCTriggerNumbering::ringFromTriggerLabels(_station, cscid) != 1))
	  {
	    result = thechamber->layer(CSCConstants::KEY_ALCT_LAYER)->centerOfWireGroup(wire_group).eta();
	  }
	else {
	  const unsigned nStrips = layerGeom->numberOfStrips();
	  const unsigned nStripsPerBin = CSCConstants::MAX_NUM_STRIPS/numBins;
	  /**
	   * Calculate Eta correction
	   */

	  // Check that no strips will be left out.
	  if (nStrips%numBins != 0 || CSCConstants::MAX_NUM_STRIPS%numBins != 0)
	    edm::LogWarning("CSCSectorReceiverLUT")
	      << "getGlobalEtaValue warning: number of strips "
	      << nStrips << " (" << CSCConstants::MAX_NUM_STRIPS
	      << ") is not divisible by numBins " << numBins
	      << " Station " << _station << " sector " << _sector
	      << " subsector " << _subsector << " cscid " << cscid << "\n";

	  unsigned    maxStripPrevBin = 0, maxStripThisBin = 0;
	  unsigned    correctionStrip;
	  LocalPoint  lPoint;
	  GlobalPoint gPoint;
	  // Bins phi_local and find the the middle strip for each bin.
	  maxStripThisBin = nStripsPerBin * (phi_local+1);
	  if (maxStripThisBin <= nStrips) {
	    correctionStrip = nStripsPerBin/2 * (2*phi_local+1);
	  }
	  else {
	    // If the actual number of strips in the chamber is smaller than
	    // the number of strips corresponding to the right edge of this phi
	    // local bin, we take the middle strip between number of strips
	    // at the left edge of the bin and the actual number of strips.
	    maxStripPrevBin = nStripsPerBin * phi_local;
	    correctionStrip = (nStrips+maxStripPrevBin)/2;
	  }

	  lPoint = layerGeom->stripWireGroupIntersection(correctionStrip, wire_group);
	  gPoint = thechamber->layer(CSCConstants::KEY_ALCT_LAYER)->surface().toGlobal(lPoint);

	  // end calc of eta correction.
	  result = gPoint.eta();
	}
      }
    }
  catch (cms::Exception &e)
    {
      LogDebug("CSCSectorReceiver|OutofBoundInput") << e.what();
    }

  return std::fabs(result);
}


gbletadat CSCSectorReceiverLUT::calcGlobalEtaME(const gbletaadd& address) const
{
  gbletadat result;
  double float_eta = getGlobalEtaValue(address.cscid, address.wire_group, address.phi_local);
  unsigned int_eta = 0;
  unsigned bend_global = 0; // not filled yet... will change when it is.
  const double etaPerBin = (CSCTFConstants::maxEta - CSCTFConstants::minEta)/CSCTFConstants::etaBins;
  const unsigned me12EtaCut = 56;

  if ((float_eta < CSCTFConstants::minEta) || (float_eta >= CSCTFConstants::maxEta))
    {
      edm::LogWarning("CSCSectorReceiverLUT:OutOfBounds")
	<< "CSCSectorReceiverLUT warning: float_eta = " << float_eta
	<< " minEta = " << CSCTFConstants::minEta << " maxEta = " << CSCTFConstants::maxEta
	<< "   station " << _station << " sector " << _sector
	<< " chamber "   << address.cscid << " wire group " << address.wire_group;

      throw cms::Exception("CSCSectorReceiverLUT")
	<< "+++ Value of CSC ID, " << float_eta
	<< ", is out of bounds [" << CSCTFConstants::minEta << "-"
	<< CSCTFConstants::maxEta << ") +++\n";

      //if (float_eta < CSCTFConstants::minEta)
      //result.global_eta = 0;
      //else if (float_eta >= CSCTFConstants::maxEta)
      //result.global_eta = CSCTFConstants::etaBins - 1;
    }
  else
    {
      float_eta -= CSCTFConstants::minEta;
      float_eta = float_eta/etaPerBin;
      int_eta = static_cast<unsigned>(float_eta);
      /* Commented until I find out its use.
      // Fine-tune eta boundary between DT and CSC.
      if ((intEta == L1MuCSCSetup::CscEtaStart() && (L1MuCSCSetup::CscEtaStartCorr() > 0.) ) ||
	  (intEta == L1MuCSCSetup::CscEtaStart() - 1 && (L1MuCSCSetup::CscEtaStartCorr() < 0.) ) ) {
	bitEta = (thisEta-minEta-L1MuCSCSetup::CscEtaStartCorr())/EtaPerBin;
	intEta = static_cast<int>(bitEta);
      }
      */
      if (_station == 1 && address.cscid >= static_cast<unsigned>(CSCTriggerNumbering::minTriggerCscId())
	  && address.cscid <= static_cast<unsigned>(CSCTriggerNumbering::maxTriggerCscId()) )
	{
	  unsigned ring = CSCTriggerNumbering::ringFromTriggerLabels(_station, address.cscid);

	  if      (ring == 1 && int_eta <  me12EtaCut) {int_eta = me12EtaCut;}
	  else if (ring == 2 && int_eta >= me12EtaCut) {int_eta = me12EtaCut-1;}
	}
      result.global_eta = int_eta;
    }
  result.global_bend = bend_global;

  return result;
}

gbletadat CSCSectorReceiverLUT::globalEtaME(int tphi_bend, int tphi_local, int twire_group, int tcscid) const
{
  gbletadat result;
  gbletaadd theadd;

  theadd.phi_bend = tphi_bend;
  theadd.phi_local = (tphi_local>>(CSCBitWidths::kLocalPhiDataBitWidth - 2)) & 0x3; // want 2 msb of local phi
  theadd.wire_group = twire_group;
  theadd.cscid = tcscid;

  if(useMiniLUTs && isTMB07) result = CSCSectorReceiverMiniLUT::calcGlobalEtaMEMini(_endcap, _sector, _station, _subsector, theadd.toint());
  else if(LUTsFromFile) result = me_global_eta[theadd.toint()];
  else result = calcGlobalEtaME(theadd);

  return result;
}

gbletadat CSCSectorReceiverLUT::globalEtaME(unsigned address) const
{
  gbletadat result;
  gbletaadd theadd(address);

  if(useMiniLUTs && isTMB07) result = CSCSectorReceiverMiniLUT::calcGlobalEtaMEMini(_endcap, _sector, _station, _subsector, address);
  else if(LUTsFromFile) result = me_global_eta[address];
  else result = calcGlobalEtaME(theadd);
  return result;
}

gbletadat CSCSectorReceiverLUT::globalEtaME(gbletaadd address) const
{
  gbletadat result;

  if(useMiniLUTs && isTMB07) result = CSCSectorReceiverMiniLUT::calcGlobalEtaMEMini(_endcap, _sector, _station, _subsector, address.toint());
  else if(LUTsFromFile) result = me_global_eta[address.toint()];
  else result = calcGlobalEtaME(address);
  return result;
}

std::string CSCSectorReceiverLUT::encodeFileIndex() const {
  std::string fileName = "";
  if (_station == 1) {
    if (_subsector == 1) fileName += "1a";
    if (_subsector == 2) fileName += "1b";
  }
  else if (_station == 2) fileName += "2";
  else if (_station == 3) fileName += "3";
  else if (_station == 4) fileName += "4";
  fileName += "End";
  if (_endcap == 1) fileName += "1";
  else                fileName += "2";
  fileName += "Sec";
  if      (_sector == 1) fileName += "1";
  else if (_sector == 2) fileName += "2";
  else if (_sector == 3) fileName += "3";
  else if (_sector == 4) fileName += "4";
  else if (_sector == 5) fileName += "5";
  else if (_sector == 6) fileName += "6";
  fileName += "LUT";
  return fileName;
}

void CSCSectorReceiverLUT::readLUTsFromFile()
{
  if(!me_lcl_phi_loaded)
    {
      me_lcl_phi = new lclphidat[1<<CSCBitWidths::kLocalPhiAddressWidth];
      memset(me_lcl_phi, 0, (1<<CSCBitWidths::kLocalPhiAddressWidth)*sizeof(short));
      std::string fName(me_lcl_phi_file.fullPath());
      std::ifstream LocalPhiLUT;

      edm::LogInfo("CSCSectorReceiverLUT") << "Loading SR LUT: " << fName;

      if(isBinary)
	{
	  LocalPhiLUT.open(fName.c_str(),std::ios::binary);
          LocalPhiLUT.seekg(0,std::ios::end);
          int length = LocalPhiLUT.tellg();
          if(length == (1<<CSCBitWidths::kLocalPhiAddressWidth)*sizeof(short))
	    {
	      LocalPhiLUT.seekg(0,std::ios::beg);
	      LocalPhiLUT.read(reinterpret_cast<char*>(me_lcl_phi),length);
	      LocalPhiLUT.close();
	    }
	  else
	    edm::LogError("CSCSectorReceiverLUT") << "File "<< fName << " is incorrect size!";
	  LocalPhiLUT.close();
	}
      else
        {
          LocalPhiLUT.open(fName.c_str());
      	  unsigned i = 0;
	  unsigned short temp = 0;
          while(!LocalPhiLUT.eof() && i < 1<<CSCBitWidths::kLocalPhiAddressWidth)
	    {
	      LocalPhiLUT >> temp;
	      me_lcl_phi[i++] = (*reinterpret_cast<lclphidat*>(&temp));
	    }
	  LocalPhiLUT.close();
	}
    }
  if(!me_global_phi)
    {
      me_global_phi = new gblphidat[1<<CSCBitWidths::kGlobalPhiAddressWidth];
      memset(me_global_phi, 0, (1<<CSCBitWidths::kGlobalPhiAddressWidth)*sizeof(short));
      std::string fName = me_gbl_phi_file.fullPath();
      std::ifstream GlobalPhiLUT;

      edm::LogInfo("CSCSectorReceiverLUT") << "Loading SR LUT: " << fName;

      if(isBinary)
        {
          GlobalPhiLUT.open(fName.c_str(),std::ios::binary);
          GlobalPhiLUT.seekg(0,std::ios::end);
          int length = GlobalPhiLUT.tellg();
          if(length == (1<<CSCBitWidths::kGlobalPhiAddressWidth)*sizeof(short))
            {
              GlobalPhiLUT.seekg(0,std::ios::beg);
              GlobalPhiLUT.read(reinterpret_cast<char*>(me_global_phi),length);
            }
          else
            edm::LogError("CSCSectorReceiverLUT") << "File "<< fName << " is incorrect size!";
          GlobalPhiLUT.close();
        }
      else
        {
          GlobalPhiLUT.open( fName.c_str());
          unsigned short temp = 0;
          unsigned i = 0;
          while(!GlobalPhiLUT.eof() && i < 1<<CSCBitWidths::kGlobalPhiAddressWidth)
	    {
	      GlobalPhiLUT >> temp;
	      me_global_phi[i++] = (*reinterpret_cast<gblphidat*>(&temp));
	    }
          GlobalPhiLUT.close();
        }
    }
  if(!mb_global_phi && _station == 1) // MB lut only in station one.
    {
      mb_global_phi = new gblphidat[1<<CSCBitWidths::kGlobalPhiAddressWidth];
      memset(mb_global_phi, 0, (1<<CSCBitWidths::kGlobalPhiAddressWidth)*sizeof(short));
      std::string fName = mb_gbl_phi_file.fullPath();
      std::ifstream GlobalPhiLUT;

      edm::LogInfo("CSCSectorReceiverLUT") << "Loading SR LUT: " << fName;

      if(isBinary)
        {
          GlobalPhiLUT.open( fName.c_str(),std::ios::binary);
          GlobalPhiLUT.seekg(0,std::ios::end);
          int length = GlobalPhiLUT.tellg();
          if(length == (1<<CSCBitWidths::kGlobalPhiAddressWidth)*sizeof(short))
            {
              GlobalPhiLUT.seekg(0,std::ios::beg);
              GlobalPhiLUT.read(reinterpret_cast<char*>(mb_global_phi),length);
            }
          else
            edm::LogError("CSCSectorReceiverLUT") << "File "<< fName << " is incorrect size!";
          GlobalPhiLUT.close();
        }
      else
        {
          GlobalPhiLUT.open(fName.c_str());
          unsigned short temp = 0;
          unsigned i = 0;
          while(!GlobalPhiLUT.eof() && i < 1<<CSCBitWidths::kGlobalPhiAddressWidth)
	    {
	      GlobalPhiLUT >> temp;
	      mb_global_phi[i++] = (*reinterpret_cast<gblphidat*>(&temp));
	    }
          GlobalPhiLUT.close();
        }
    }
  if(!me_global_eta)
    {
      me_global_eta = new gbletadat[1<<CSCBitWidths::kGlobalEtaAddressWidth];
      memset(me_global_eta, 0, (1<<CSCBitWidths::kGlobalEtaAddressWidth)*sizeof(short));
      std::string fName = me_gbl_eta_file.fullPath();
      std::ifstream GlobalEtaLUT;

      edm::LogInfo("CSCSectorReceiverLUT") << "Loading SR LUT: " << fName;

      if(isBinary)
	{
	  GlobalEtaLUT.open(fName.c_str(),std::ios::binary);
	  GlobalEtaLUT.seekg(0,std::ios::end);
	  int length = GlobalEtaLUT.tellg();
	  if(length == (1<<CSCBitWidths::kGlobalEtaAddressWidth)*sizeof(short))
	    {
	      GlobalEtaLUT.seekg(0,std::ios::beg);
	      GlobalEtaLUT.read(reinterpret_cast<char*>(me_global_eta),length);
	    }
	  else
	    edm::LogError("CSCSectorReceiverLUT") << "File "<< fName << " is incorrect size!";
	  GlobalEtaLUT.close();
	}
      else
	{
	  GlobalEtaLUT.open(fName.c_str());
	  unsigned short temp = 0;
	  unsigned i = 0;
	  while(!GlobalEtaLUT.eof() && i < 1<<CSCBitWidths::kGlobalEtaAddressWidth)
	  {
	    GlobalEtaLUT >> temp;
	    me_global_eta[i++] = (*reinterpret_cast<gbletadat*>(&temp));
	  }
	  GlobalEtaLUT.close();
	}
    }
}

