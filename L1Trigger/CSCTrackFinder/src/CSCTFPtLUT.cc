#include <L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCTFConstants.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <fstream>

ptdat* CSCTFPtLUT::pt_lut = NULL;
bool CSCTFPtLUT::lut_read_in = false;
L1MuTriggerScales CSCTFPtLUT::trigger_scale;
CSCTFPtMethods CSCTFPtLUT::ptMethods;

///KK
#include "CondFormats/L1TObjects/interface/L1MuCSCPtLut.h"
#include "CondFormats/DataRecord/interface/L1MuCSCPtLutRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h>
CSCTFPtLUT::CSCTFPtLUT(const edm::EventSetup& es){
	pt_method = 1;
	lowQualityFlag = 4;
	pt_lut = new ptdat[1<<21];

	edm::ESHandle<L1MuCSCPtLut> ptLUT;
	es.get<L1MuCSCPtLutRcd>().get(ptLUT);
	const L1MuCSCPtLut *myConfigPt_ = ptLUT.product();

	memcpy((void*)pt_lut,(void*)myConfigPt_->lut(),(1<<21)*sizeof(ptdat));

	lut_read_in = true;
}
///

CSCTFPtLUT::CSCTFPtLUT(const edm::ParameterSet& pset)
{
  read_pt_lut = pset.getUntrackedParameter<bool>("ReadPtLUT",false);
  if(read_pt_lut)
    {
      pt_lut_file = pset.getUntrackedParameter<edm::FileInPath>("PtLUTFile",edm::FileInPath("L1Trigger/CSCTrackFinder/LUTs/L1CSCPtLUT.dat"));
      isBinary = pset.getUntrackedParameter<bool>("isBinary", false);
    }

  // Determine the pt assignment method to use
  // 1 - Darin's parameterization method
  // 2 - Cathy Yeh's chi-square minimization method
  // 3 - Hybrid
  pt_method = pset.getUntrackedParameter<unsigned>("PtMethod",1);
  // what does this mean???
  lowQualityFlag = pset.getUntrackedParameter<unsigned>("LowQualityFlag",4);

  if(read_pt_lut && !lut_read_in)
    {
      pt_lut = new ptdat[1<<21];
      readLUT();
      lut_read_in = true;
    }
}

ptdat CSCTFPtLUT::Pt(const ptadd& address) const
{
  ptdat result;
  if(read_pt_lut)
    result = pt_lut[address.toint()];
  else
    result = calcPt(address);
  return result;
}

ptdat CSCTFPtLUT::Pt(const unsigned& address) const
{
  return Pt(ptadd(address));
}

ptdat CSCTFPtLUT::Pt(const unsigned& delta_phi_12, const unsigned& delta_phi_23,
		     const unsigned& track_eta, const unsigned& track_mode,
		     const unsigned& track_fr, const unsigned& delta_phi_sign) const
{
  ptadd address;
  address.delta_phi_12 = delta_phi_12;
  address.delta_phi_23 = delta_phi_23;
  address.track_eta = track_eta;
  address.track_mode = track_mode;
  address.track_fr = track_fr;
  address.delta_phi_sign = delta_phi_sign;

  return Pt(address);
}

ptdat CSCTFPtLUT::Pt(const unsigned& delta_phi_12, const unsigned& track_eta,
		     const unsigned& track_mode, const unsigned& track_fr,
		     const unsigned& delta_phi_sign) const
{
  ptadd address;
  address.delta_phi_12 = ((1<<8)-1)&delta_phi_12;
  address.delta_phi_23 = ((1<<4)-1)&(delta_phi_12>>8);
  address.track_eta = track_eta;
  address.track_mode = track_mode;
  address.track_fr = track_fr;
  address.delta_phi_sign = delta_phi_sign;

  return Pt(address);
}

// Taken from spbits.h :
#define MODE_ACC 15 // mode for accelerator tracks

ptdat CSCTFPtLUT::calcPt(const ptadd& address) const
{
  ptdat result;

  float etaR = 0, ptR_front = 0, ptR_rear = 0, dphi12R = 0, dphi23R = 0;
  int charge12, charge23;
  unsigned type, mode, eta, fr, quality, charge, absPhi12, absPhi23;

  eta = address.track_eta;
  mode = address.track_mode;
  fr = address.track_fr;
  charge = address.delta_phi_sign;
  quality = trackQuality(eta, mode);
  unsigned front_pt, rear_pt;
  unsigned front_quality, rear_quality;

  etaR = trigger_scale.getRegionalEtaScale(2)->getLowEdge(2*eta+1);

  front_quality = rear_quality = quality;

  //  kluge to use 2-stn track in overlap region
  //  see also where this routine is called, and encode LUTaddress, and assignPT
  if ((mode == 2 || mode == 3 || mode == 4) && (eta<3)) mode = 6;
  if ((mode == 5)                           && (eta<3)) mode = 8;

  switch(mode)
    {
    case 2:
    case 3:
    case 4:
    case 5:
      type = mode - 1;
      charge12 = 1;
      absPhi12 = address.delta_phi_12;
      absPhi23 = address.delta_phi_23;

      if(charge) charge23 = 1;
      else charge23 = -1;

      // now convert to real numbers for input into PT assignment algos.

      if(pt_method == 1) // param method
	{
	  dphi12R = (static_cast<float>(absPhi12<<1)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
	  dphi23R = (static_cast<float>(absPhi23<<4)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
	  if(charge12 * charge23 < 0) dphi23R = -dphi23R;

	  ptR_front = ptMethods.Pt3Stn(type, etaR, dphi12R, dphi23R, 1);
	  ptR_rear  = ptMethods.Pt3Stn(type, etaR, dphi12R, dphi23R, 0);

	}
      else if(pt_method == 2) // cathy's method
	{
	  if(type <= 2)
	    {
	      ptR_front = ptMethods.Pt3StnChiSq(type+3, etaR, absPhi12<<1, ((charge == 0) ? -(absPhi23<<4) : (absPhi23<<4)), 1);
	      ptR_rear  = ptMethods.Pt3StnChiSq(type+3, etaR, absPhi12<<1, ((charge == 0) ? -(absPhi23<<4) : (absPhi23<<4)), 0);
	    }
	  else
	    {
	      ptR_front = ptMethods.Pt2StnChiSq(type-2, etaR, absPhi12<<1, 1);
	      ptR_rear  = ptMethods.Pt2StnChiSq(type-2, etaR, absPhi12<<1, 0);
	    }

	}
      else // hybrid
	{

	  if(type <= 2)
	    {
	      ptR_front = ptMethods.Pt3StnHybrid(type+3, etaR, absPhi12<<1, ((charge == 0) ? -(absPhi23<<4) : (absPhi23<<4)), 1);
	      ptR_rear  = ptMethods.Pt3StnHybrid(type+3, etaR, absPhi12<<1, ((charge == 0) ? -(absPhi23<<4) : (absPhi23<<4)), 0);
	    }
	  else
	    {
	      ptR_front = ptMethods.Pt2StnHybrid(type-2, etaR, absPhi12<<1, 1);
	      ptR_rear  = ptMethods.Pt2StnHybrid(type-2, etaR, absPhi12<<1, 0);
	    }

	}
      break;
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
      type = mode - 5;

      if(charge) absPhi12 = address.delta_phi();
      else
	{
	  int temp_phi = address.delta_phi();
	  absPhi12 = static_cast<unsigned>(-temp_phi) & 0xfff;
	}

      if(absPhi12 < (1<<9))
	{
	  if(pt_method == 1 || type == 5)
	    {
	      dphi12R = (static_cast<float>(absPhi12)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;

	      ptR_front = ptMethods.Pt2Stn(type, etaR, dphi12R, 1);
	      ptR_rear  = ptMethods.Pt2Stn(type, etaR, dphi12R, 0);

	    }
	  else if(pt_method == 2)
	    {
	      ptR_front = ptMethods.Pt2StnChiSq(type-1, etaR, absPhi12, 1);
	      ptR_rear  = ptMethods.Pt2StnChiSq(type-1, etaR, absPhi12, 0);
	    }
	  else
	    {
	      ptR_front = ptMethods.Pt2StnHybrid(type-1, etaR, absPhi12, 1);
	      ptR_rear  = ptMethods.Pt2StnHybrid(type-1, etaR, absPhi12, 0);
	    }
	}
      else
	{
	  ptR_front = trigger_scale.getPtScale()->getLowEdge(1);
	  ptR_rear  = trigger_scale.getPtScale()->getLowEdge(1);
	}
      break;
    case 11:
    case 12:
    case 14:
      type = 2;

      if(charge) absPhi12 = address.delta_phi();
      else
	{
	  int temp_phi = address.delta_phi();
	  absPhi12 = static_cast<unsigned>(-temp_phi) & 0xfff;
	}
      if(absPhi12 < (1<<9))
	{
	  dphi12R = (static_cast<float>(absPhi12)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
	  ptR_front = ptMethods.Pt2Stn(type, etaR, dphi12R, 1);
	  ptR_rear  = ptMethods.Pt2Stn(type, etaR, dphi12R, 0);
	}
      else
	{
	  ptR_front = trigger_scale.getPtScale()->getLowEdge(1);
	  ptR_rear  = trigger_scale.getPtScale()->getLowEdge(1);
	}
      break;
    case 13:
    case 15:
      type = 4;

      if(charge) absPhi12 = address.delta_phi();
      else
	{
	  int temp_phi = address.delta_phi();
	  absPhi12 = static_cast<unsigned>(-temp_phi) & 0xfff;
	}
      if(absPhi12 < (1<<9))
	{
	  dphi12R = (static_cast<float>(absPhi12)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
	  ptR_front = ptMethods.Pt2Stn(type, etaR, dphi12R, 1);
	  ptR_rear  = ptMethods.Pt2Stn(type, etaR, dphi12R, 0);
	}
      else
	{
	  ptR_front = trigger_scale.getPtScale()->getLowEdge(1);
	  ptR_rear  = trigger_scale.getPtScale()->getLowEdge(1);
	}
      break;
    case 1:
      ptR_front = trigger_scale.getPtScale()->getLowEdge(5);
      ptR_rear  = trigger_scale.getPtScale()->getLowEdge(5);
      break;
    default: // Tracks in this category are not considered muons.
      ptR_front = trigger_scale.getPtScale()->getLowEdge(0);
      ptR_rear  = trigger_scale.getPtScale()->getLowEdge(0);
    };

  front_pt = trigger_scale.getPtScale()->getPacked(ptR_front);
  rear_pt  = trigger_scale.getPtScale()->getPacked(ptR_rear);

  // kluge to set arbitrary Pt for some tracks with lousy resolution (and no param)
  if ((front_pt==0 || front_pt==1) && (eta<3) && quality==1 && pt_method != 2) front_pt = 31;
  if ((rear_pt==0  || rear_pt==1) && (eta<3) && quality==1 && pt_method != 2) rear_pt = 31;

  if(pt_method != 2 && quality == 1)
    {
      if (front_pt < 5) front_pt = 5;
      if (rear_pt  < 5) rear_pt  = 5;
    }

  if( mode==MODE_ACC ){ // halo muon track:
    result.front_rank = 1;
    result.rear_rank  = 1;
  } else { // any other tracks:
    result.front_rank = front_pt | front_quality << 5;
    result.rear_rank  = rear_pt  | rear_quality << 5;
  }
  result.charge_valid_front = 1; //ptMethods.chargeValid(front_pt, quality, eta, pt_method);
  result.charge_valid_rear  = 1; //ptMethods.chargeValid(rear_pt, quality, eta, pt_method);

  return result;
}

#undef MODE_ACC

unsigned CSCTFPtLUT::trackQuality(const unsigned& eta, const unsigned& mode) const
{
 // eta and mode should be only 4-bits, since that is the input to the large LUT
    if (eta>15 || mode>15)
      {
        //std::cout << "Error: Eta or Mode out of range in AU quality assignment" << std::endl;
        edm::LogError("CSCTFPtLUT::trackQuality()")<<"Eta or Mode out of range in AU quality assignment";
        return 0;
      }
    unsigned int quality;
    switch (mode) {
    case 2:
      quality = 3;
      break;
    case 3:
      quality = 3;
      break;
    case 4:
      /// DEA try increasing quality
      //        quality = 2;
      quality = 3;
      break;
    case 5:
      quality = 1;
      break;
    case 6:
      if (eta>=3)
	quality = 2;
      else
	quality = 1;
      break;
    case 7:
      quality = 2;
      break;
    case 8:
      quality = 1;
      break;
    case 9:
      quality = 1;
      break;
    case 10:
      quality = 1;
      break;
    case 11:
      quality = 3;
      break;
    case 12:
      quality = 3;
      break;
    case 13:
      quality = 2;
      break;
    case 14:
      quality = 2;
      break;
    case 15:
      quality = 2;
      break;
      //DEA: keep muons that fail delta phi cut
    case 1:
      quality = 1;
    default:
      quality = 0;
      break;
    }

    // allow quality = 1 only in overlap region or eta = 1.6 region
    //    if ((quality == 1) && (eta >= 4) && (eta != 6) && (eta != 7)) quality = 0;
    //    if ( (quality == 1) && (eta >= 4) ) quality = 0;

    if ( (quality == 1) && (eta >= 4) && (eta < 11)
	 && ((lowQualityFlag&4)==0) ) quality = 0;
    if ( (quality == 1) && (eta < 4) && ((lowQualityFlag&1)==0)
	 && ((lowQualityFlag&4)==0) ) quality = 0;
    if ( (quality == 1) && (eta >=11) && ((lowQualityFlag&2)==0)
	 && ((lowQualityFlag&4)==0) ) quality = 0;

    return quality;
}

void CSCTFPtLUT::readLUT()
{
  std::ifstream PtLUT;

  if(isBinary)
    {
      PtLUT.open(pt_lut_file.fullPath().c_str(), std::ios::binary);
      PtLUT.seekg(0, std::ios::end);
      int length = PtLUT.tellg();;
      if( length == (1<<CSCBitWidths::kPtAddressWidth)*sizeof(short) )
	{
	  PtLUT.seekg(0, std::ios::beg);
	  PtLUT.read(reinterpret_cast<char*>(pt_lut),length);
	}
      else
	{
	  edm::LogError("CSCPtLUT") << "File " << pt_lut_file.fullPath() << " is incorrect size!\n";
	}
      PtLUT.close();
    }
  else
    {
      PtLUT.open(pt_lut_file.fullPath().c_str());
      unsigned i = 0;
      unsigned short temp = 0;
      while(!PtLUT.eof() && i < 1 << CSCBitWidths::kPtAddressWidth)
	{
	  PtLUT >> temp;
	  pt_lut[i++] = (*reinterpret_cast<ptdat*>(&temp));
	}
      PtLUT.close();
    }
}


