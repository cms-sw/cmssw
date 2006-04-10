#include <L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeomManager.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCPatternLUT.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCFrontRearLUT.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCBitWidths.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>

#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <Geometry/Vector/interface/LocalPoint.h>
#include <Geometry/Vector/interface/GlobalPoint.h>

#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <fstream>

CSCSectorReceiverLUT::lclphidat* CSCSectorReceiverLUT::me_lcl_phi = NULL;
bool CSCSectorReceiverLUT::me_lcl_phi_loaded = false;

CSCSectorReceiverLUT::CSCSectorReceiverLUT(int endcap, int sector, int subsector, int station,
					   const edm::ParameterSet & pset):_endcap(endcap),_sector(sector),
									   _subsector(subsector),
									   _station(station)
{
  LUTsFromFile = pset.getUntrackedParameter<bool>("ReadLUTs",false);
  isBinary = pset.getUntrackedParameter<bool>("Binary",false);
  lut_path = pset.getUntrackedParameter<std::string>("LUTPath","./");
  me_global_eta = NULL;
  if(LUTsFromFile) readLUTsFromFile();
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

  return localPhi(theadd);
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

double CSCSectorReceiverLUT::getEtaValue(const unsigned& thecscid, const unsigned& thewire_group, const unsigned& thephi_local) const
{
  double result = 0.0;
  unsigned wire_group = thewire_group;
  int cscid = thecscid;
  unsigned phi_local = thephi_local;

  if(cscid < CSCTriggerNumbering::minTriggerCscId() || cscid > CSCTriggerNumbering::maxTriggerCscId())
    {
      LogDebug("CSCSectorReceiverLUT|getEtaValue") << " warning: cscId " << cscid
						   << " is out of bounds (1-" << CSCTriggerNumbering::maxTriggerCscId();
      cscid = CSCTriggerNumbering::maxTriggerCscId();
    }

  CSCTriggerGeomManager* thegeom = CSCTriggerGeometry::get();
  CSCLayerGeometry* layerGeom = NULL;
  const unsigned numBins = 1 << 2; // 4 local phi bins
  
  if(phi_local > numBins - 1)
    {
      LogDebug("CSCSectorReceiverLUT|getEtaValue") << "warning: phiL " << phi_local
						   << " is out of bounds (0-" << numBins - 1;
      phi_local = numBins - 1;
    }
  try 
    {    
      const CSCChamber* thechamber = thegeom->chamber(_endcap,_station,_sector,_subsector,cscid);     
      if(thechamber) 
	{
	  if(_station != 1 && CSCTriggerNumbering::ringFromTriggerLabels(_station, cscid) != 1)
	    {
	      layerGeom = const_cast<CSCLayerGeometry*>(thechamber->layer(3)->geometry());
	      const unsigned nWireGroups = layerGeom->numberOfStrips();

	      if(wire_group > nWireGroups)
		{
		  LogDebug("CSCSectorReceiverLUT|getEtaValue") << "warning: wireGroup "
                                                               << wire_group << " is out of bounds (0-"
                                                               << nWireGroups;
                  wire_group = nWireGroups - 1;
		}

	      result = thechamber->layer(3)->centerOfWireGroup(wire_group+1).eta();
	    }
	  else
	    {
	      layerGeom = const_cast<CSCLayerGeometry*>(thechamber->layer(3)->geometry());
	      
	      const unsigned nStrips = layerGeom->numberOfStrips();
	      const unsigned nWireGroups = layerGeom->numberOfWireGroups();
	      const unsigned nStripsPerBin = CSCConstants::MAX_NUM_STRIPS/numBins;
	      
	      
	      if(wire_group > nWireGroups) // apply maximum limit
		{
		  LogDebug("CSCSectorReceiverLUT|getEtaValue") << "warning: wireGroup "
							       << wire_group << " is out of bounds (0-"
							       << nWireGroups;
		  wire_group = nWireGroups - 1;
		}
	      
	      /**
	       * Calculate Eta correction
	       */
	      
	      // Check that no strips will be left out.
	      
	      if (nStrips%numBins != 0 || CSCConstants::MAX_NUM_STRIPS%numBins != 0)
		LogDebug("CSCSectorReceiverLUT|EtaCorrectionWarning") << "calcEtaCorrection warning: number of strips "
								      << nStrips << " (" << CSCConstants::MAX_NUM_STRIPS
								      << ") is not divisible by numBins " << numBins
								      << " Station " << _station << " sector " << _sector
								      << " subsector " << _subsector << " cscid " << cscid;
	      
	      unsigned    maxStripPrevBin = 0, maxStripThisBin = 0;
	      unsigned    correctionStrip;
	      LocalPoint  lPoint;
	      GlobalPoint gPoint;
	      // Bins phi_local and find the the middle strip for each bin.
	      maxStripPrevBin = nStripsPerBin * (phi_local);
	      maxStripThisBin = nStripsPerBin * (phi_local+1);
	      if (maxStripThisBin <= nStrips) 
		{
		  correctionStrip = nStripsPerBin/2 * (2*phi_local+1);
		  maxStripPrevBin = maxStripThisBin;
		}
	      else 
		{
		  // If the actual number of strips in the chamber is smaller than
		  // the number of strips corresponding to the right edge of this phi
		  // local bin, we take the middle strip between number of strips
		  // at the left edge of the bin and the actual number of strips.
		  correctionStrip = (nStrips+maxStripPrevBin)/2;
		}
	      
	      lPoint = layerGeom->stripWireGroupIntersection(correctionStrip, wire_group+1);
	      if(thechamber) gPoint = thechamber->layer(3)->surface().toGlobal(lPoint);
	      
	      // end calc of eta correction.
	      result = gPoint.eta();
	    }
	}
    }
  catch (cms::Exception &e)
    {
      LogDebug("CSCSectorReceiver:OutofBoundInput") << e.what();
    }
  
  return std::fabs(result);
}


CSCSectorReceiverLUT::gbletadat CSCSectorReceiverLUT::calcGlobalEtaME(const gbletaadd& address) const
{
  gbletadat result;
  double float_eta = getEtaValue(address.cscid, address.wire_group, address.phi_local);
  unsigned int_eta = 0;
  unsigned bend_global = 0; // not filled yet... will change when it is.
  const double etaPerBin = (CSCConstants::maxEta - CSCConstants::minEta)/CSCConstants::etaBins;
  const unsigned me12EtaCut = 56;
    
  if ((float_eta < CSCConstants::minEta) || (float_eta >= CSCConstants::maxEta)) 
    {
      
      LogDebug("CSCSectorReceiverLUT:OutOfBounds") << "L1MuCSCSectorReceiverLUT warning: float_eta = " << float_eta
						   << " minEta = " << CSCConstants::minEta << " maxEta = " << CSCConstants::maxEta
						   << "   station " << _station << " sector " << _sector
						   << " chamber "   << address.cscid << " wire group " << address.wire_group;
      
      if (float_eta < CSCConstants::minEta) 
	result.global_eta = 0;
      else if (float_eta >= CSCConstants::maxEta) 
	result.global_eta = CSCConstants::etaBins - 1;
    }
  else
    {  
      float_eta -= CSCConstants::minEta;
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
      if (_station == 1 && address.cscid > CSCTriggerNumbering::minTriggerCscId() 
	  && address.cscid < CSCTriggerNumbering::maxTriggerCscId() )
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

CSCSectorReceiverLUT::gbletadat CSCSectorReceiverLUT::globalEtaME(int tphi_bend, int tphi_local, int twire_group, int tcscid) const
{
  gbletadat result;
  gbletaadd theadd;

  theadd.phi_bend = tphi_bend;
  theadd.phi_local = (tphi_local>>(CSCBitWidths::kLocalPhiDataBitWidth - 2)) & 0x3; // want 2 msb of local phi
  theadd.wire_group = twire_group;
  theadd.cscid = tcscid;
  
  if(LUTsFromFile) result = me_global_eta[(*reinterpret_cast<unsigned*>(&theadd))];
  else result = calcGlobalEtaME(theadd);

  //  if(address.wire_group == 0 && address.phi_local==0) std::cout << result.global_eta << std::endl;

  return result;
}

CSCSectorReceiverLUT::gbletadat CSCSectorReceiverLUT::globalEtaME(unsigned address) const
{
  gbletadat result;
  if(LUTsFromFile) result = me_global_eta[address];
  else result = calcGlobalEtaME(*reinterpret_cast<gbletaadd*>(&address));
  return result;
}

CSCSectorReceiverLUT::gbletadat CSCSectorReceiverLUT::globalEtaME(gbletaadd address) const
{
  gbletadat result;
  if(LUTsFromFile) result = me_global_eta[(*reinterpret_cast<unsigned*>(&address))];
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
      std::string fName("LocalPhiLUT");
      std::ifstream LocalPhiLUT;

      edm::LogInfo("CSCSectorReceiverLUT|loadLUT") << "Loading SR LUT: " << fName; 

      fName += ((isBinary) ? ".bin" : ".dat");

      if(isBinary)
	{
	  LocalPhiLUT.open((lut_path+fName).c_str(),std::ios::binary);
          LocalPhiLUT.seekg(0,std::ios::end);
          int length = LocalPhiLUT.tellg();
          if(length == (1<<CSCBitWidths::kLocalPhiAddressWidth)*sizeof(short))
	    {
	      LocalPhiLUT.seekg(0,std::ios::beg);
	      LocalPhiLUT.read(reinterpret_cast<char*>(me_lcl_phi),length);
	      LocalPhiLUT.close();
	    }
	  else
	    edm::LogError("CSCSectorReceiverLUT|loadLUT") << "File "<< fName << " is incorrect size!";
	  LocalPhiLUT.close();
	}
      else
        {
          LocalPhiLUT.open((lut_path+fName).c_str());
      	  unsigned i = 0;
	  unsigned short temp = 0;
          while(!LocalPhiLUT.eof() && i < 1<<CSCBitWidths::kLocalPhiAddressWidth)
	    {
	      LocalPhiLUT >> temp;
	      memcpy(me_lcl_phi + i++, &temp, sizeof(short));
	    }
	  LocalPhiLUT.close();
	}
    }
  if(!me_global_eta) 
    {
      me_global_eta = new gbletadat[1<<CSCBitWidths::kGlobalEtaAddressWidth];
      memset(me_global_eta, 0, (1<<CSCBitWidths::kGlobalEtaAddressWidth)*sizeof(short));
      std::string fName = std::string("GlobalEtaME") + encodeFileIndex();
      std::ifstream GlobalEtaLUT;
      
      edm::LogInfo("CSCSectorReceiverLUT|loadLUT") << "Loading SR LUT: " << fName;

      fName += ((isBinary) ? ".bin" : ".dat");
      std::cout<<(lut_path+fName) << std::endl;

      if(isBinary)
	{
	  GlobalEtaLUT.open((lut_path + fName).c_str(),std::ios::binary);
	  GlobalEtaLUT.seekg(0,std::ios::end);
	  int length = GlobalEtaLUT.tellg();
	  if(length == (1<<CSCBitWidths::kGlobalEtaAddressWidth)*sizeof(short))
	    {
	      GlobalEtaLUT.seekg(0,std::ios::beg);
	      GlobalEtaLUT.read(reinterpret_cast<char*>(me_global_eta),length);
	    }
	  else
	    edm::LogError("CSCSectorReceiverLUT|loadLUT") << "File "<< fName << " is incorrect size!";
	  GlobalEtaLUT.close();
	}
      else
	{
	  GlobalEtaLUT.open((lut_path + fName).c_str());
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

