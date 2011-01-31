/*****************************************************
 * 28/01/2010
 * GP: added new switch to use the beam start Pt LUTs
 * if (eta > 2.1) 2 stations tracks have quality 2
 *                3 stations tracks have quality 3
 * NB: no matter if the have ME1
 * 
 * --> by default is set to true
 *****************************************************/

#include <L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCTFConstants.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <fstream>

ptdat* CSCTFPtLUT::pt_lut = NULL;
bool CSCTFPtLUT::lut_read_in = false;
// L1MuTriggerScales CSCTFPtLUT::trigger_scale;
// L1MuTriggerPtScale CSCTFPtLUT::trigger_ptscale;
// CSCTFPtMethods CSCTFPtLUT::ptMethods;

///KK
#include "CondFormats/L1TObjects/interface/L1MuCSCPtLut.h"
#include "CondFormats/DataRecord/interface/L1MuCSCPtLutRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h>

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"

// info for getPtScale() pt scale in GeV
// low edges of pt bins
/*     const float ptscale[33] = {  */
/*       -1.,   0.0,   1.5,   2.0,   2.5,   3.0,   3.5,   4.0, */
/*       4.5,   5.0,   6.0,   7.0,   8.0,  10.0,  12.0,  14.0,   */
/*       16.0,  18.0,  20.0,  25.0,  30.0,  35.0,  40.0,  45.0,  */
/*       50.0,  60.0,  70.0,  80.0,  90.0, 100.0, 120.0, 140.0, 1.E6 }; */


CSCTFPtLUT::CSCTFPtLUT(const edm::EventSetup& es) 
    : read_pt_lut(false),
      isBinary(false)
{
	pt_method = 4;
        //std::cout << "pt_method from 4 " << std::endl; 
	lowQualityFlag = 4;
	isBeamStartConf = true;
	pt_lut = new ptdat[1<<21];

	edm::ESHandle<L1MuCSCPtLut> ptLUT;
	es.get<L1MuCSCPtLutRcd>().get(ptLUT);
	const L1MuCSCPtLut *myConfigPt_ = ptLUT.product();

	memcpy((void*)pt_lut,(void*)myConfigPt_->lut(),(1<<21)*sizeof(ptdat));

	lut_read_in = true;

	edm::ESHandle< L1MuTriggerScales > scales ;
	es.get< L1MuTriggerScalesRcd >().get( scales ) ;
	trigger_scale = scales.product() ;

	edm::ESHandle< L1MuTriggerPtScale > ptScale ;
	es.get< L1MuTriggerPtScaleRcd >().get( ptScale ) ;
	trigger_ptscale = ptScale.product() ;

	ptMethods = CSCTFPtMethods( ptScale.product() ) ;
}
///

CSCTFPtLUT::CSCTFPtLUT(const edm::ParameterSet& pset,
		       const L1MuTriggerScales* scales,
		       const L1MuTriggerPtScale* ptScale)
  : trigger_scale( scales ),
    trigger_ptscale( ptScale ),
    ptMethods( ptScale ),
    read_pt_lut(false),
    isBinary(false)
{
  //read_pt_lut = pset.getUntrackedParameter<bool>("ReadPtLUT",false);
  read_pt_lut = pset.getParameter<bool>("ReadPtLUT");
  if(read_pt_lut)
    {
      pt_lut_file = pset.getParameter<edm::FileInPath>("PtLUTFile");
      //isBinary = pset.getUntrackedParameter<bool>("isBinary", false);
      isBinary = pset.getParameter<bool>("isBinary");

      edm::LogInfo("CSCTFPtLUT::CSCTFPtLUT") << "Reading file: "
					     << pt_lut_file.fullPath().c_str()
					     << " isBinary?(1/0): "
					     << isBinary;
    }

  // Determine the pt assignment method to use
  // 1 - Darin's parameterization method
  // 2 - Cathy Yeh's chi-square minimization method
  // 3 - Hybrid
  // 4 - Anna's parameterization method
  // 5 - Anna's parameterization method 
         //with improvments at ME1/1a: find max pt for 3 links hypothesis
  // 11 - Anna's: for fw 20101011 <- 2011 data taking <- not valide any more
  // 12 - Anna's: for fw 20101011 <- 2011 data taking <- not valide any more
          //with improvments at ME1/1a: find max pt for 3 links hypothesis
  // 21 - Anna's: for fw 20110118 and up, curves with data 2010 <- 2011 data taking
  // 22 - Anna's: for fw 20110118 and up, curves with data 2010 <- 2011 data taking
          //with improvments at ME1/1a: find max pt for 3 links hypothesis
  // 23 - Anna's: for fw 20110118 and up, curves with MC like method 4 <- 2011 data taking
  // 24 - Anna's: for fw 20110118 and up, curves with MC like method 4 <- 2011 data taking
          //with improvments at ME1/1a: find max pt for 3 links hypothesis
  pt_method = pset.getUntrackedParameter<unsigned>("PtMethod",4);
  //std::cout << "pt_method from pset " << std::endl; 
  // what does this mean???
  lowQualityFlag = pset.getUntrackedParameter<unsigned>("LowQualityFlag",4);

  if(read_pt_lut && !lut_read_in)
    {
      pt_lut = new ptdat[1<<21];
      readLUT();
      lut_read_in = true;
    }

  isBeamStartConf = pset.getUntrackedParameter<bool>("isBeamStartConf", true);
}

ptdat CSCTFPtLUT::Pt(const ptadd& address) const
{
  ptdat result;
  if(read_pt_lut)
  {
    int shortAdd = (address.toint()& 0x1fffff);
    result = pt_lut[shortAdd];
  } else
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


ptdat CSCTFPtLUT::calcPt(const ptadd& address) const
{
  ptdat result;

  double Pi  = acos(-1.);
  float etaR = 0, ptR_front = 0, ptR_rear = 0, dphi12R = 0, dphi23R = 0;
  int charge12, charge23;
  unsigned type, mode, eta, fr, quality, charge, absPhi12, absPhi23;

  eta = address.track_eta;
  mode = address.track_mode;
  fr = address.track_fr;
  charge = address.delta_phi_sign;
  quality = trackQuality(eta, mode, fr);
  unsigned front_pt, rear_pt;
  front_pt = 0.; rear_pt = 0.;
  unsigned front_quality, rear_quality;

  etaR = trigger_scale->getRegionalEtaScale(2)->getLowEdge(2*eta+1);

  front_quality = rear_quality = quality;


//***************************************************//
  if(pt_method == 23 || pt_method == 24){ //here we have only pt_methods greater then 
                       //for fw 20110118 <- 2011 data taking, curves from MC like method 4
  // mode definition you could find at page 6 & 7: 
  // http://www.phys.ufl.edu/~madorsky/sp/2011-11-18/sp_core_interface.pdf 
  // it is valid starting the beggining of 2011 
  switch(mode)
    {
    case 2:
    case 3:
    case 4:
    case 5:

      charge12 = 1;
      absPhi12 = address.delta_phi_12;
      absPhi23 = address.delta_phi_23;

      if(charge) charge23 = 1;
      else charge23 = -1;

      dphi12R = (static_cast<float>(absPhi12<<1)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
      dphi23R = (static_cast<float>(absPhi23<<4)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
      if(charge12 * charge23 < 0) dphi23R = -dphi23R;

      ptR_front = ptMethods.Pt3Stn2010(int(mode), etaR, dphi12R, dphi23R, 1, int(pt_method));
      ptR_rear  = ptMethods.Pt3Stn2010(int(mode), etaR, dphi12R, dphi23R, 0, int(pt_method));    

      if(pt_method == 24 && mode != 5 && etaR > 2.1)//exclude mode without ME11a
        {
            float dphi12Rmin = dphi12R - Pi*10/180/3; // 10/3 degrees 
            float dphi12Rmax = dphi12R + Pi*10/180/3; // 10/3 degrees
            float dphi23Rmin = dphi23R;
            float dphi23Rmax = dphi23R;
            //if(dphi12Rmin*dphi12R < 0) dphi23Rmin = -dphi23R;
            //if(dphi12Rmax*dphi12R < 0) dphi23Rmax = -dphi23R;
            float ptR_front_min = ptMethods.Pt3Stn2010(int(mode), etaR, dphi12Rmin, dphi23Rmin, 1, int(pt_method));
            float ptR_rear_min = ptMethods.Pt3Stn2010(int(mode), etaR, dphi12Rmin, dphi23Rmin, 0, int(pt_method));
            float ptR_front_max = ptMethods.Pt3Stn2010(int(mode), etaR, dphi12Rmax, dphi23Rmax, 1, int(pt_method));
            float ptR_rear_max = ptMethods.Pt3Stn2010(int(mode), etaR, dphi12Rmax, dphi23Rmax, 0, int(pt_method));
            // select max pt solution for 3 links:
            ptR_front = std::max(ptR_front, ptR_front_min);
            ptR_front = std::max(ptR_front, ptR_front_max);
            ptR_rear = std::max(ptR_rear, ptR_rear_min);
            ptR_rear = std::max(ptR_rear, ptR_rear_max);
        }
      break;
    case 6: // for mode 6, 7 and 13 add CLCT information in dph23 bit and iME11 in charge bit  
    case 7:
    case 13: // ME1-ME4
      int iME11;
      iME11 = int(charge); // = 0 if ME1/1, = 1 if ME1/2 or ME1/3 station 
      if(iME11 == 1 && etaR > 1.6) etaR = 1.55; // shift for ME1/2 station  
      if(iME11 == 0 && etaR < 1.6) etaR = 1.65; // shift for ME1/1 station 
      absPhi12 = address.delta_phi_12;
      //int CLCT_pattern = static_cast<int>(address.delta_phi_23);
      int CLCT_pattern;
      CLCT_pattern = int(address.delta_phi_23);

      dphi12R = (static_cast<float>(absPhi12<<1)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;

      //std::cout<< " Sector_rad = " << (CSCTFConstants::SECTOR_RAD) << std::endl;
      ptR_front = ptMethods.Pt2Stn2010(int(mode), etaR, dphi12R, 1, int(pt_method));
      ptR_rear  = ptMethods.Pt2Stn2010(int(mode), etaR, dphi12R, 0, int(pt_method));
      if((pt_method == 24) && etaR > 2.1)//exclude tracks without ME11a 
        {
           float dphi12Rmin = fabs(fabs(dphi12R) - Pi*10/180/3); // 10/3 degrees 
           float ptR_front_min = ptMethods.Pt2Stn2010(int(mode), etaR, dphi12Rmin, 1, int(pt_method));
           float ptR_rear_min = ptMethods.Pt2Stn2010(int(mode), etaR, dphi12Rmin, 0, int(pt_method));
           // select max pt solution for 3 links:
           ptR_front = std::max(ptR_front, ptR_front_min);
           ptR_rear = std::max(ptR_rear, ptR_rear_min);
        }
      if((CLCT_pattern =! 8) && (CLCT_pattern =! 9) && (CLCT_pattern != 10) && (ptR_front > 5.)) ptR_front = 5.;
      if((CLCT_pattern =! 8) && (CLCT_pattern =! 9) && (CLCT_pattern != 10) && (ptR_rear > 5.)) ptR_rear = 5.;

      break;
    case 8:
    case 9:
    case 10:
      if(charge) absPhi12 = address.delta_phi();
      else
	{
	  int temp_phi = address.delta_phi();
	  absPhi12 = static_cast<unsigned>(-temp_phi) & 0xfff;
	}

      dphi12R = (static_cast<float>(absPhi12)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;

      //std::cout<< " Sector_rad = " << (CSCTFConstants::SECTOR_RAD) << std::endl;
      ptR_front = ptMethods.Pt2Stn2010(int(mode), etaR, dphi12R, 1, int(pt_method));
      ptR_rear  = ptMethods.Pt2Stn2010(int(mode), etaR, dphi12R, 0, int(pt_method));

      break;
    // for overlap DT-CSC region using curves from data 2010
    case 11: // FR = 1 -> b1-1-3,     FR = 0 -> b1-3 
    case 12: // FR = 1 -> b1-2-3,     FR = 0 -> b1-2 
    case 14: // FR = 1 -> b1-1-2-(3), FR = 0 -> b1-1

        charge12 = 1;
        absPhi12 = address.delta_phi_12;
        absPhi23 = address.delta_phi_23;

        if(charge) charge23 = 1;
        else charge23 = -1;

        dphi12R = (static_cast<float>(absPhi12<<1)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
        dphi23R = float(absPhi23);
        if(charge12 * charge23 < 0) dphi23R = -dphi23R;

        int mode1;
        mode1 = int(mode);
        if(fr == 1 && mode1 == 11) mode1 = 14; // 3 station track we use dphi12 and phiBend for 2 and 3 station track

        ptR_front = ptMethods.Pt3Stn2011(mode1, etaR, dphi12R, dphi23R, int(0), int(pt_method));
        ptR_rear = ptMethods.Pt3Stn2011(mode1, etaR, dphi12R, dphi23R, int(0), int(pt_method));

      break;
    case 15: // halo trigger
    case 1: // tracks that fail delta phi cuts
      ptR_front = trigger_ptscale->getPtScale()->getLowEdge(3); // 2 GeV
      ptR_rear  = trigger_ptscale->getPtScale()->getLowEdge(3); 
      break;
    default: // Tracks in this category are not considered muons.
      ptR_front = trigger_ptscale->getPtScale()->getLowEdge(0); // 0 GeV 
      ptR_rear  = trigger_ptscale->getPtScale()->getLowEdge(0);
    };// end switch

  front_pt = trigger_ptscale->getPtScale()->getPacked(ptR_front);
  rear_pt  = trigger_ptscale->getPtScale()->getPacked(ptR_rear);

  } //end pt_methods 23 & 24 

//***************************************************//
//***************************************************//
  if(pt_method == 21 || pt_method == 22){ //here we have only pt_methods greater then 
                       //for fw 20110118 <- 2011 data taking
  // mode definition you could find at page 6 & 7: 
  // http://www.phys.ufl.edu/~madorsky/sp/2011-11-18/sp_core_interface.pdf 
  // it is valid starting the beggining of 2011 
  switch(mode)
    {
    case 2:
    case 3:
    case 4:
    case 5:

      charge12 = 1;
      absPhi12 = address.delta_phi_12;
      absPhi23 = address.delta_phi_23;

      if(charge) charge23 = 1;
      else charge23 = -1;

      dphi12R = (static_cast<float>(absPhi12<<1)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
      dphi23R = (static_cast<float>(absPhi23<<4)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
      if(charge12 * charge23 < 0) dphi23R = -dphi23R;

      ptR_front = ptMethods.Pt3Stn2011(int(mode), etaR, dphi12R, dphi23R, 1, int(pt_method));
      ptR_rear  = ptMethods.Pt3Stn2011(int(mode), etaR, dphi12R, dphi23R, 0, int(pt_method));    

      if(pt_method == 22 && mode != 5 && etaR > 2.1)//exclude mode without ME11a
        {
            float dphi12Rmin = dphi12R - Pi*10/180/3; // 10/3 degrees 
            float dphi12Rmax = dphi12R + Pi*10/180/3; // 10/3 degrees
            float dphi23Rmin = dphi23R;
            float dphi23Rmax = dphi23R;
            //if(dphi12Rmin*dphi12R < 0) dphi23Rmin = -dphi23R;
            //if(dphi12Rmax*dphi12R < 0) dphi23Rmax = -dphi23R;
            float ptR_front_min = ptMethods.Pt3Stn2011(int(mode), etaR, dphi12Rmin, dphi23Rmin, 1, int(pt_method));
            float ptR_rear_min = ptMethods.Pt3Stn2011(int(mode), etaR, dphi12Rmin, dphi23Rmin, 0, int(pt_method));
            float ptR_front_max = ptMethods.Pt3Stn2011(int(mode), etaR, dphi12Rmax, dphi23Rmax, 1, int(pt_method));
            float ptR_rear_max = ptMethods.Pt3Stn2011(int(mode), etaR, dphi12Rmax, dphi23Rmax, 0, int(pt_method));
            // select max pt solution for 3 links:
            ptR_front = std::max(ptR_front, ptR_front_min);
            ptR_front = std::max(ptR_front, ptR_front_max);
            ptR_rear = std::max(ptR_rear, ptR_rear_min);
            ptR_rear = std::max(ptR_rear, ptR_rear_max);
        }
      break;
    case 6: // for mode 6, 7 and 13 add CLCT information in dph23 bit and iME11 in charge bit  
    case 7:
    case 13: // ME1-ME4
      int iME11;
      iME11 = int(charge);
      absPhi12 = address.delta_phi_12;
      //int CLCT_pattern = static_cast<int>(address.delta_phi_23);
      int CLCT_pattern;
      CLCT_pattern = int(address.delta_phi_23);

      dphi12R = (static_cast<float>(absPhi12<<1)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;

      //std::cout<< " Sector_rad = " << (CSCTFConstants::SECTOR_RAD) << std::endl;
      ptR_front = ptMethods.Pt2Stn2011(int(mode), etaR, dphi12R, 1, int(pt_method), iME11);
      ptR_rear  = ptMethods.Pt2Stn2011(int(mode), etaR, dphi12R, 0, int(pt_method), iME11);
      if((pt_method == 22) && etaR > 2.1)//exclude tracks without ME11a 
        {
           float dphi12Rmin = fabs(fabs(dphi12R) - Pi*10/180/3); // 10/3 degrees 
           float ptR_front_min = ptMethods.Pt2Stn2011(int(mode), etaR, dphi12Rmin, 1, int(pt_method), iME11);
           float ptR_rear_min = ptMethods.Pt2Stn2011(int(mode), etaR, dphi12Rmin, 0, int(pt_method), iME11);
           // select max pt solution for 3 links:
           ptR_front = std::max(ptR_front, ptR_front_min);
           ptR_rear = std::max(ptR_rear, ptR_rear_min);
        }
      if((CLCT_pattern =! 8) && (CLCT_pattern =! 9) && (CLCT_pattern != 10) && (ptR_front > 5.)) ptR_front = 5.;
      if((CLCT_pattern =! 8) && (CLCT_pattern =! 9) && (CLCT_pattern != 10) && (ptR_rear > 5.)) ptR_rear = 5.;

      break;
    case 8:
    case 9:
    case 10:
      if(charge) absPhi12 = address.delta_phi();
      else
	{
	  int temp_phi = address.delta_phi();
	  absPhi12 = static_cast<unsigned>(-temp_phi) & 0xfff;
	}

      dphi12R = (static_cast<float>(absPhi12)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;

      //std::cout<< " Sector_rad = " << (CSCTFConstants::SECTOR_RAD) << std::endl;
      ptR_front = ptMethods.Pt2Stn2011(int(mode), etaR, dphi12R, 1, int(pt_method), int(2));
      ptR_rear  = ptMethods.Pt2Stn2011(int(mode), etaR, dphi12R, 0, int(pt_method), int(2));

      break;
    case 11: // FR = 1 -> b1-1-3,     FR = 0 -> b1-3 
    case 12: // FR = 1 -> b1-2-3,     FR = 0 -> b1-2 
    case 14: // FR = 1 -> b1-1-2-(3), FR = 0 -> b1-1

        charge12 = 1;
        absPhi12 = address.delta_phi_12;
        absPhi23 = address.delta_phi_23;

        if(charge) charge23 = 1;
        else charge23 = -1;

        dphi12R = (static_cast<float>(absPhi12<<1)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
        dphi23R = float(absPhi23);
        if(charge12 * charge23 < 0) dphi23R = -dphi23R;

        int mode1;
        mode1 = int(mode);
        if(fr == 1 && mode1 == 11) mode1 = 14; // 3 station track we use dphi12 and phiBend for 2 and 3 station track

        ptR_front = ptMethods.Pt3Stn2011(mode1, etaR, dphi12R, dphi23R, int(0), int(pt_method));
        ptR_rear = ptMethods.Pt3Stn2011(mode1, etaR, dphi12R, dphi23R, int(0), int(pt_method));

      break;
    case 15: // halo trigger
    case 1: // tracks that fail delta phi cuts
      ptR_front = trigger_ptscale->getPtScale()->getLowEdge(3); // 2 GeV
      ptR_rear  = trigger_ptscale->getPtScale()->getLowEdge(3); 
      break;
    default: // Tracks in this category are not considered muons.
      ptR_front = trigger_ptscale->getPtScale()->getLowEdge(0); // 0 GeV 
      ptR_rear  = trigger_ptscale->getPtScale()->getLowEdge(0);
    };// end switch

  front_pt = trigger_ptscale->getPtScale()->getPacked(ptR_front);
  rear_pt  = trigger_ptscale->getPtScale()->getPacked(ptR_rear);

  } //end pt_methods greater or equal to 21 

//***************************************************//
//***************************************************//
  if(pt_method >= 11 && pt_method < 20){ //here we have only pt_methods greater or equal to 11 
                       //for fw 20101011 <- 2011 data taking
  // mode definition you could find at page 6 & 7: 
  // http://www.phys.ufl.edu/~madorsky/sp/2010-10-11/sp_core_interface.pdf 
  // it is valid starting the beggining of 2011 
  switch(mode)
    {
    case 2:
    case 3:
    case 4:
    case 5:

      charge12 = 1;
      absPhi12 = address.delta_phi_12;
      absPhi23 = address.delta_phi_23;

      if(charge) charge23 = 1;
      else charge23 = -1;

      dphi12R = (static_cast<float>(absPhi12<<1)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
      dphi23R = (static_cast<float>(absPhi23<<4)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
      if(charge12 * charge23 < 0) dphi23R = -dphi23R;

      ptR_front = ptMethods.Pt3Stn2010(mode, etaR, dphi12R, dphi23R, 1, int(pt_method));
      ptR_rear  = ptMethods.Pt3Stn2010(mode, etaR, dphi12R, dphi23R, 0, int(pt_method));    

      if(pt_method == 12 && mode != 5 && etaR > 2.1)//exclude mode without ME11a
        {
            float dphi12Rmin = dphi12R - Pi*10/180/3; // 10/3 degrees 
            float dphi12Rmax = dphi12R + Pi*10/180/3; // 10/3 degrees
            float dphi23Rmin = dphi23R;
            float dphi23Rmax = dphi23R;
            if(dphi12Rmin*dphi12R < 0) dphi23Rmin = -dphi23R;
            if(dphi12Rmax*dphi12R < 0) dphi23Rmax = -dphi23R;
            float ptR_front_min = ptMethods.Pt3Stn2010(mode, etaR, dphi12Rmin, dphi23Rmin, 1, int(pt_method));
            float ptR_rear_min = ptMethods.Pt3Stn2010(mode, etaR, dphi12Rmin, dphi23Rmin, 0, int(pt_method));
            float ptR_front_max = ptMethods.Pt3Stn2010(mode, etaR, dphi12Rmax, dphi23Rmax, 1, int(pt_method));
            float ptR_rear_max = ptMethods.Pt3Stn2010(mode, etaR, dphi12Rmax, dphi23Rmax, 0, int(pt_method));
            // select max pt solution for 3 links:
            ptR_front = std::max(ptR_front, ptR_front_min);
            ptR_front = std::max(ptR_front, ptR_front_max);
            ptR_rear = std::max(ptR_rear, ptR_rear_min);
            ptR_rear = std::max(ptR_rear, ptR_rear_max);
        }
      break;
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
    case 13: // ME1-ME4
      type = mode - 5;

      if(charge) absPhi12 = address.delta_phi();
      else
	{
	  int temp_phi = address.delta_phi();
	  absPhi12 = static_cast<unsigned>(-temp_phi) & 0xfff;
	}

      dphi12R = (static_cast<float>(absPhi12)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;

      //std::cout<< " Sector_rad = " << (CSCTFConstants::SECTOR_RAD) << std::endl;
      ptR_front = ptMethods.Pt2Stn2010(mode, etaR, dphi12R, 1, int(pt_method));
      ptR_rear  = ptMethods.Pt2Stn2010(mode, etaR, dphi12R, 0, int(pt_method));
      if((pt_method == 12) && etaR > 2.1 && mode != 8 && mode !=9 && mode !=10)//exclude tracks without ME11a 
        {
           float dphi12Rmin = fabs(fabs(dphi12R) - Pi*10/180/3); // 10/3 degrees 
           float ptR_front_min = ptMethods.Pt2Stn2010(mode, etaR, dphi12Rmin, 1, int(pt_method));
           float ptR_rear_min = ptMethods.Pt2Stn2010(mode, etaR, dphi12Rmin, 0, int(pt_method));
           // select max pt solution for 3 links:
           ptR_front = std::max(ptR_front, ptR_front_min);
           ptR_rear = std::max(ptR_rear, ptR_rear_min);
        }

      break;
    case 11: // FR = 1 -> b1-1-3,     FR = 0 -> b1-3 
    case 12: // FR = 1 -> b1-2-3,     FR = 0 -> b1-2 
    case 14: // FR = 1 -> b1-1-2-(3), FR = 0 -> b1-1

      if(fr == 0){ // 2 station track
        if(charge) absPhi12 = address.delta_phi();
        else
          {
	    int temp_phi = address.delta_phi();
	    absPhi12 = static_cast<unsigned>(-temp_phi) & 0xfff;
          }
          dphi12R = (static_cast<float>(absPhi12)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
          ptR_rear  = ptMethods.Pt2Stn2010(mode, etaR, dphi12R, 0, int(pt_method));

      }// end fr == 0
      if(fr == 1){ // 3 station track
        charge12 = 1;
        absPhi12 = address.delta_phi_12;
        absPhi23 = address.delta_phi_23;

        if(charge) charge23 = 1;
        else charge23 = -1;

        dphi12R = (static_cast<float>(absPhi12<<1)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
        dphi23R = (static_cast<float>(absPhi23<<4)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
        if(charge12 * charge23 < 0) dphi23R = -dphi23R;

        ptR_front = ptMethods.Pt3Stn2010(mode, etaR, dphi12R, dphi23R, 1, int(pt_method));

        if(pt_method == 12 && mode != 5 && etaR > 2.1)//exclude mode without ME11a
          {
              float dphi12Rmin = dphi12R - Pi*10/180/3; // 10/3 degrees 
              float dphi12Rmax = dphi12R + Pi*10/180/3; // 10/3 degrees
              float dphi23Rmin = dphi23R;
              float dphi23Rmax = dphi23R;
              if(dphi12Rmin*dphi12R < 0) dphi23Rmin = -dphi23R;
              if(dphi12Rmax*dphi12R < 0) dphi23Rmax = -dphi23R;
              float ptR_front_min = ptMethods.Pt3Stn2010(mode, etaR, dphi12Rmin, dphi23Rmin, 1, int(pt_method));
              float ptR_front_max = ptMethods.Pt3Stn2010(mode, etaR, dphi12Rmax, dphi23Rmax, 1, int(pt_method));
              // select max pt solution for 3 links:
              ptR_front = std::max(ptR_front, ptR_front_min);
              ptR_front = std::max(ptR_front, ptR_front_max);
          }
      } // end fr == 1 

      break;
    case 15: // halo trigger
    case 1: // tracks that fail delta phi cuts
      ptR_front = trigger_ptscale->getPtScale()->getLowEdge(3); // 2 GeV
      ptR_rear  = trigger_ptscale->getPtScale()->getLowEdge(3); 
      break;
    default: // Tracks in this category are not considered muons.
      ptR_front = trigger_ptscale->getPtScale()->getLowEdge(0); // 0 GeV 
      ptR_rear  = trigger_ptscale->getPtScale()->getLowEdge(0);
    };// end switch

  front_pt = trigger_ptscale->getPtScale()->getPacked(ptR_front);
  rear_pt  = trigger_ptscale->getPtScale()->getPacked(ptR_rear);

  } //end pt_methods greater or equal to 11 
//***************************************************//
  if(pt_method <= 5){ //here we have only pt_methods less or equal to 5
  // mode definition you could find at https://twiki.cern.ch/twiki/pub/Main/PtLUTs/mode_codes.xls
  // it is valid till the end 2010 

  //  kluge to use 2-stn track in overlap region
  //  see also where this routine is called, and encode LUTaddress, and assignPT
  if (pt_method != 4 && pt_method !=5 
      && (mode == 2 || mode == 3 || mode == 4) && (eta<3)) mode = 6;
  if (pt_method != 4 && pt_method !=5 && (mode == 5)
      && (eta<3)) mode = 8;

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

      if(pt_method == 4 || pt_method == 5) // param method 2010
        {
          dphi12R = (static_cast<float>(absPhi12<<1)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
          dphi23R = (static_cast<float>(absPhi23<<4)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
          if(charge12 * charge23 < 0) dphi23R = -dphi23R;

          ptR_front = ptMethods.Pt3Stn2010(mode, etaR, dphi12R, dphi23R, 1, int(pt_method));
          ptR_rear  = ptMethods.Pt3Stn2010(mode, etaR, dphi12R, dphi23R, 0, int(pt_method));

          if(pt_method == 5 && mode != 5 && etaR > 2.1)//exclude mode without ME11a
            {
                float dphi12Rmin = dphi12R - Pi*10/180/3; // 10/3 degrees 
                float dphi12Rmax = dphi12R + Pi*10/180/3; // 10/3 degrees
                float dphi23Rmin = dphi23R;
                float dphi23Rmax = dphi23R;
                if(dphi12Rmin*dphi12R < 0) dphi23Rmin = -dphi23R;
                if(dphi12Rmax*dphi12R < 0) dphi23Rmax = -dphi23R;
                float ptR_front_min = ptMethods.Pt3Stn2010(mode, etaR, dphi12Rmin, dphi23Rmin, 1, int(pt_method));
                float ptR_rear_min = ptMethods.Pt3Stn2010(mode, etaR, dphi12Rmin, dphi23Rmin, 0, int(pt_method));
                float ptR_front_max = ptMethods.Pt3Stn2010(mode, etaR, dphi12Rmax, dphi23Rmax, 1, int(pt_method));
                float ptR_rear_max = ptMethods.Pt3Stn2010(mode, etaR, dphi12Rmax, dphi23Rmax, 0, int(pt_method));
                // select max pt solution for 3 links:
                ptR_front = std::max(ptR_front, ptR_front_min);
                ptR_front = std::max(ptR_front, ptR_front_max);
                ptR_rear = std::max(ptR_rear, ptR_rear_min);
                ptR_rear = std::max(ptR_rear, ptR_rear_max);
            }
        }
      else if(pt_method == 1) // param method
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
	  ptR_front = trigger_ptscale->getPtScale()->getLowEdge(1);
	  ptR_rear  = trigger_ptscale->getPtScale()->getLowEdge(1);
	}
      if(pt_method == 4 || pt_method == 5) // param method 2010
        {
              dphi12R = (static_cast<float>(absPhi12)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;

              //std::cout<< " Sector_rad = " << (CSCTFConstants::SECTOR_RAD) << std::endl;
              ptR_front = ptMethods.Pt2Stn2010(mode, etaR, dphi12R, 1, int(pt_method));
              ptR_rear  = ptMethods.Pt2Stn2010(mode, etaR, dphi12R, 0, int(pt_method));
              if((pt_method == 5) && etaR > 2.1 && mode != 8 && mode !=9 && mode !=10)//exclude tracks without ME11a 
                {
                   float dphi12Rmin = fabs(fabs(dphi12R) - Pi*10/180/3); // 10/3 degrees 
                   float ptR_front_min = ptMethods.Pt2Stn2010(mode, etaR, dphi12Rmin, 1, int(pt_method));
                   float ptR_rear_min = ptMethods.Pt2Stn2010(mode, etaR, dphi12Rmin, 0, int(pt_method));
                   // select max pt solution for 3 links:
                   ptR_front = std::max(ptR_front, ptR_front_min);
                   ptR_rear = std::max(ptR_rear, ptR_rear_min);
                }
        }

      break;
    case 12:  // 1-2-b1 calculated only delta_phi12 = 2-b1
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
	  ptR_front = trigger_ptscale->getPtScale()->getLowEdge(1);
	  ptR_rear  = trigger_ptscale->getPtScale()->getLowEdge(1);
	}
      if(pt_method == 4 || pt_method == 5) // param method 2010 
        {
              dphi12R = (static_cast<float>(absPhi12)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;

              ptR_front = ptMethods.Pt2Stn2010(mode, etaR, dphi12R, 1, int(pt_method));
              ptR_rear  = ptMethods.Pt2Stn2010(mode, etaR, dphi12R, 0, int(pt_method));

              if(fabs(dphi12R)<0.01 && (ptR_rear < 10 || ptR_front < 10))
                std::cout << "dphi12R = " << dphi12R << " ptR_rear = " << ptR_rear
                << " ptR_front = " << ptR_front << " etaR = " << etaR << " mode = " << mode << std::endl;
        }
      break;
    case 13:
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
	  ptR_front = trigger_ptscale->getPtScale()->getLowEdge(1);
	  ptR_rear  = trigger_ptscale->getPtScale()->getLowEdge(1);
	}

      if(pt_method == 4 || pt_method == 5) // param method 2010
        {
              dphi12R = (static_cast<float>(absPhi12)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;

              ptR_front = ptMethods.Pt2Stn2010(mode, etaR, dphi12R, 1, int(pt_method));
              ptR_rear  = ptMethods.Pt2Stn2010(mode, etaR, dphi12R, 0, int(pt_method));
              if((pt_method == 5) && etaR > 2.1)//mode = 13: ME1-ME4 exclude tracks without ME11a 
                {
                   float dphi12Rmin = fabs(fabs(dphi12R) - Pi*10/180/3); // 10/3 degrees 
                   float ptR_front_min = ptMethods.Pt2Stn2010(mode, etaR, dphi12Rmin, 1, int(pt_method));
                   float ptR_rear_min = ptMethods.Pt2Stn2010(mode, etaR, dphi12Rmin, 0, int(pt_method));
                   // select max pt solution for 3 links:
                   ptR_front = std::max(ptR_front, ptR_front_min);
                   ptR_rear = std::max(ptR_rear, ptR_rear_min);
                }
        }

      break;
    case 11:
      // singles trigger
      ptR_front = trigger_ptscale->getPtScale()->getLowEdge(5);
      ptR_rear  = trigger_ptscale->getPtScale()->getLowEdge(5);
      //ptR_front = trigger_ptscale->getPtScale()->getLowEdge(31);
      //ptR_rear  = trigger_ptscale->getPtScale()->getLowEdge(31);
      break;
    case 15:
      // halo trigger
      ptR_front = trigger_ptscale->getPtScale()->getLowEdge(5);
      ptR_rear  = trigger_ptscale->getPtScale()->getLowEdge(5);
      break;
    case 1:
      // tracks that fail delta phi cuts
      ptR_front = trigger_ptscale->getPtScale()->getLowEdge(5);
      ptR_rear  = trigger_ptscale->getPtScale()->getLowEdge(5); 
     break;
    default: // Tracks in this category are not considered muons.
      ptR_front = trigger_ptscale->getPtScale()->getLowEdge(0);
      ptR_rear  = trigger_ptscale->getPtScale()->getLowEdge(0);
    };

  front_pt = trigger_ptscale->getPtScale()->getPacked(ptR_front);
  rear_pt  = trigger_ptscale->getPtScale()->getPacked(ptR_rear);

  // kluge to set arbitrary Pt for some tracks with lousy resolution (and no param)
  if(pt_method != 4 && pt_method != 5) 
    {
      if ((front_pt==0 || front_pt==1) && (eta<3) && quality==1 && pt_method != 2) front_pt = 31;
      if ((rear_pt==0  || rear_pt==1) && (eta<3) && quality==1 && pt_method != 2) rear_pt = 31;
    }
  if(pt_method != 2 && pt_method != 4 && quality == 1)
    {
      if (front_pt < 5) front_pt = 5;
      if (rear_pt  < 5) rear_pt  = 5;
    }

  // in order to match the pt assignement of the previous routine
  if(isBeamStartConf && pt_method != 2 && pt_method != 4 && pt_method !=5) {
    if(quality == 3 && mode == 5) {
      
      if (front_pt < 5) front_pt = 5;
      if (rear_pt  < 5) rear_pt  = 5;
    }

    if(quality == 2 && mode > 7 && mode < 11) {
      
      if (front_pt < 5) front_pt = 5;
      if (rear_pt  < 5) rear_pt  = 5;
    }
  }

  } // end if for pt_method less or equal to 5
//***************************************************//

 
  result.front_rank = front_pt | front_quality << 5;
  result.rear_rank  = rear_pt  | rear_quality << 5;

  result.charge_valid_front = 1; //ptMethods.chargeValid(front_pt, quality, eta, pt_method);
  result.charge_valid_rear  = 1; //ptMethods.chargeValid(rear_pt, quality, eta, pt_method);


  /*  if (mode == 1) { 
    std::cout << "F_pt: "      << front_pt      << std::endl;
    std::cout << "R_pt: "      << rear_pt       << std::endl;
    std::cout << "F_quality: " << front_quality << std::endl;
    std::cout << "R_quality: " << rear_quality  << std::endl;
    std::cout << "F_rank: " << std::hex << result.front_rank << std::endl;
    std::cout << "R_rank: " << std::hex << result.rear_rank  << std::endl;
  }
  */
  return result;
}


unsigned CSCTFPtLUT::trackQuality(const unsigned& eta, const unsigned& mode, const unsigned& fr) const
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
      if(pt_method > 10 && eta < 3) quality = 1; //eta < 1.2  
      break;
    case 3:
    case 4:
      /// DEA try increasing quality
      //        quality = 2;
      quality = 3;
      break;
    case 5:
      quality = 1;
      if (isBeamStartConf && eta >= 12 && pt_method < 20) // eta > 2.1
	quality = 3;
      break;
    case 6:
      if (eta>=3) // eta > 1.2
	quality = 2;
      else
	quality = 1;
      break;
    case 7:
      quality = 2;
      if(pt_method > 10 && eta < 3) quality = 1; //eta < 1.2  
      break;
    case 8:
      quality = 1;
      if (isBeamStartConf && eta >= 12 && pt_method < 20) // eta > 2.1
	quality = 2;
      break;
    case 9:
      quality = 1;
      if (isBeamStartConf && eta >= 12 && pt_method < 20) // eta > 2.1
	quality = 2;
      break;
    case 10:
      quality = 1;
      if (isBeamStartConf && eta >= 12 && pt_method < 20) // eta > 2.1
	quality = 2;
      break;
    case 11:
      // single LCTs
      quality = 1;
      // overlap region
      if(pt_method > 10 && fr == 0) quality = 2;
      if(pt_method > 10 && fr == 1) quality = 3;
      if(pt_method > 20 && fr == 0) quality = 3;
      break;
    case 12:
      quality = 3;
      // overlap region
      if(pt_method > 10 && fr == 0) quality = 2;
      if(pt_method > 10 && fr == 1) quality = 3;
      if(pt_method > 20 && fr == 0) quality = 3;
      break;
    case 13:
      quality = 2;
      break;
    case 14:
      quality = 2;
      // overlap region
      if(pt_method > 10 && fr == 0) quality = 2;
      if(pt_method > 10 && fr == 1) quality = 3;
      if(pt_method > 20 && fr == 0) quality = 3;
      break;
    case 15:
      // halo triggers
      quality = 1;
      break;
      //DEA: keep muons that fail delta phi cut
    case 1:
      quality = 1;
      break;
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


