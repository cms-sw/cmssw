#include <L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeomManager.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCPatternLUT.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCFrontRearLUT.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCBitWidths.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>

CSCSectorReceiverLUT::lclphidat* CSCSectorReceiverLUT::me_lcl_phi = NULL;
bool CSCSectorReceiverLUT::me_lcl_phi_loaded = false;

CSCSectorReceiverLUT::CSCSectorReceiverLUT(int endcap, int sector, int subsector, int station,
					   const edm::ParameterSet & pset):_endcap(endcap),_sector(sector),
									   _subsector(subsector),
									   _station(station)
{
  LUTsFromFile = pset.getUntrackedParameter<bool>("ReadLUTs",false);
  //if(LUTsFromFile) readLUTsFromFile();
  me_global_eta = NULL;

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
}

CSCSectorReceiverLUT::lclphidat CSCSectorReceiverLUT::calcLocalPhi(const lclphiadd& theadd) const
{
  lclphidat data;

  int maxPhiL = 1<<CSCBitWidths::kLocalPhiDataBitWidth;
  double binPhiL = static_cast<double>(maxPhiL)/(2.*CSCConstants::MAX_NUM_STRIPS);

  memset(&data,0,sizeof(lclphidat));
  double patternOffset = CSCPatternLUT::getPosition(theadd.clct_pattern);
  
  if(theadd.strip < 2*CSCConstants::MAX_NUM_STRIPS)
    if(theadd.pattern_type == 1) // if halfstrip
      data.phi_local = static_cast<unsigned>((0.5 + theadd.strip + patternOffset)*binPhiL);
    else // if distrip
      data.phi_local = static_cast<unsigned>((2 + theadd.strip + 4.*patternOffset)*binPhiL);
  else // set out of bounds values
    if(theadd.pattern_type == 1)
      data.phi_local = static_cast<unsigned>((0.5 + (2*CSCConstants::MAX_NUM_STRIPS-1) + patternOffset)*binPhiL);
    else
      data.phi_local = static_cast<unsigned>((2 + (2*CSCConstants::MAX_NUM_STRIPS-1) + 4.*patternOffset)*binPhiL);
  
  /// Local Phi Bend is always zero. Until we start using it.
  data.phi_bend_local = 0;

  return data; //return LUT result
}


void CSCSectorReceiverLUT::fillLocalPhiLUT()
{ 
  // read data in from a file... Add this later.
}

CSCSectorReceiverLUT::lclphidat CSCSectorReceiverLUT::localPhi(int strip, int pattern, int quality, int lr) const
{
  lclphiadd theadd;

  theadd.strip = strip;
  theadd.clct_pattern = pattern;
  theadd.quality = quality;
  theadd.lr = lr;
  theadd.spare = 0;

  return calcLocalPhi(theadd);
}

CSCSectorReceiverLUT::lclphidat CSCSectorReceiverLUT::localPhi(unsigned address) const
{
  lclphidat result;

  if(LUTsFromFile) result = me_lcl_phi[address];
  else result = calcLocalPhi(*reinterpret_cast<lclphiadd*>(&address));

  return result;
}

CSCSectorReceiverLUT::lclphidat CSCSectorReceiverLUT::localPhi(lclphiadd address) const
{
  lclphidat result;
  
  if(LUTsFromFile) result = me_lcl_phi[(*reinterpret_cast<unsigned*>(&address))];
  else result = calcLocalPhi(address);
  
  return result;
}

double CSCSectorReceiverLUT::getEtaValue(unsigned cscid, unsigned wire_group) const
{
  double result;
  CSCTriggerGeomManager* thegeom = CSCTriggerGeometry::get();
  const CSCChamber* thechamber = thegeom->chamber(_endcap,_station,_sector,_subsector,cscid);
  
  result = thechamber->layer(3)->centerOfWireGroup(wire_group).eta();

  return result;
}

CSCSectorReceiverLUT::gbletadat CSCSectorReceiverLUT::calcGlobalEtaME(const gbletaadd& address) const
{
  gbletadat result;
  return result;
}

CSCSectorReceiverLUT::gbletadat CSCSectorReceiverLUT::globalEtaME(int phi_bend, int phi_local, int wire_group, int cscid) const
{
  gbletadat result;
  return result;
}

CSCSectorReceiverLUT::gbletadat CSCSectorReceiverLUT::globalEtaME(unsigned address) const
{
  gbletadat result;
  return result;
}

CSCSectorReceiverLUT::gbletadat CSCSectorReceiverLUT::globalEtaME(gbletaadd address) const
{
  gbletadat result;
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

  return fileName;
}

void CSCSectorReceiverLUT::readLUTsFromFile()
{
  if(!me_lcl_phi_loaded) fillLocalPhiLUT();
}

