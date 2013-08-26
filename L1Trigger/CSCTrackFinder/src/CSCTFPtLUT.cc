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

/*
  These arrays were defined such that they take integer dPhi --> dPhi-units which are used by CSCTFPtMethods.
  The values were defined using:
  
  Phi_{i} = Phi_{i-1} + MAX( phi_Unit * alphi^{i}, phi_Unit), where phi_Unit = 62 deg / 4096, and alpha is solved for
  such that the phi-space is resonably covered. This will be better optimized in the future.
  
*/

// Array that maps the 5-bit integer dPhi --> dPhi-units. It is assumed that this is used for dPhi23,
// which has a maximum value of 3.83 degrees (255 units) in the extrapolation units.
const int CSCTFPtLUT::dPhiNLBMap_5bit[32] =
     {	0	,	1	,	2	,	4	,	5	,	7	,	9	,	11	,	13	,	15	,	18	,	21	,	24	,	28	,	32	,	37	,	41	,	47	,	53	,	60	,	67	,	75	,	84	,	94	,	105	,	117	,	131	,	145	,	162	,	180	,	200	,	222};

// Array that maps the 7-bit integer dPhi --> dPhi-units. It is assumed that this is used for dPhi12,
// which has a maximum value of 7.67 degrees (511 units) in the extrapolation units.
const int CSCTFPtLUT::dPhiNLBMap_7bit[128] =
  {	0	,	1	,	2	,	3	,	4	,	5	,	6	,	8	,	9	,	10	,	11	,	12	,	14	,	15	,	16	,	17	,	19	,	20	,	21	,	23	,	24	,	26	,	27	,	29	,	30	,	32	,	33	,	35	,	37	,	38	,	40	,	42	,	44	,	45	,	47	,	49	,	51	,	53	,	55	,	57	,	59	,	61	,	63	,	65	,	67	,	70	,	72	,	74	,	77	,	79	,	81	,	84	,	86	,	89	,	92	,	94	,	97	,	100	,	103	,	105	,	108	,	111	,	114	,	117	,	121	,	124	,	127	,	130	,	134	,	137	,	141	,	144	,	148	,	151	,	155	,	159	,	163	,	167	,	171	,	175	,	179	,	183	,	188	,	192	,	197	,	201	,	206	,	210	,	215	,	220	,	225	,	230	,	235	,	241	,	246	,	251	,	257	,	263	,	268	,	274	,	280	,	286	,	292	,	299	,	305	,	312	,	318	,	325	,	332	,	339	,	346	,	353	,	361	,	368	,	376	,	383	,	391	,	399	,	408	,	416	,	425	,	433	,	442	,	451	,	460	,	469	,	479	,	489 };

// Array that maps the 8-bit integer dPhi --> dPhi-units. It is assumed that this is used for dPhi12,
// which has a maximum value of 7.67 degrees (511 units) in the extrapolation units.
const int CSCTFPtLUT::dPhiNLBMap_8bit[256] =
 {	0	,	1	,	2	,	3	,	4	,	5	,	6	,	7	,	8	,	9	,	10	,	11	,	12	,	13	,	14	,	16	,	17	,	18	,	19	,	20	,	21	,	22	,	23	,	24	,	25	,	27	,	28	,	29	,	30	,	31	,	32	,	33	,	35	,	36	,	37	,	38	,	39	,	40	,	42	,	43	,	44	,	45	,	46	,	48	,	49	,	50	,	51	,	53	,	54	,	55	,	56	,	58	,	59	,	60	,	61	,	63	,	64	,	65	,	67	,	68	,	69	,	70	,	72	,	73	,	74	,	76	,	77	,	79	,	80	,	81	,	83	,	84	,	85	,	87	,	88	,	90	,	91	,	92	,	94	,	95	,	97	,	98	,	100	,	101	,	103	,	104	,	105	,	107	,	108	,	110	,	111	,	113	,	115	,	116	,	118	,	119	,	121	,	122	,	124	,	125	,	127	,	129	,	130	,	132	,	133	,	135	,	137	,	138	,	140	,	141	,	143	,	145	,	146	,	148	,	150	,	151	,	153	,	155	,	157	,	158	,	160	,	162	,	163	,	165	,	167	,	169	,	171	,	172	,	174	,	176	,	178	,	180	,	181	,	183	,	185	,	187	,	189	,	191	,	192	,	194	,	196	,	198	,	200	,	202	,	204	,	206	,	208	,	210	,	212	,	214	,	216	,	218	,	220	,	222	,	224	,	226	,	228	,	230	,	232	,	234	,	236	,	238	,	240	,	242	,	244	,	246	,	249	,	251	,	253	,	255	,	257	,	259	,	261	,	264	,	266	,	268	,	270	,	273	,	275	,	277	,	279	,	282	,	284	,	286	,	289	,	291	,	293	,	296	,	298	,	300	,	303	,	305	,	307	,	310	,	312	,	315	,	317	,	320	,	322	,	324	,	327	,	329	,	332	,	334	,	337	,	340	,	342	,	345	,	347	,	350	,	352	,	355	,	358	,	360	,	363	,	366	,	368	,	371	,	374	,	376	,	379	,	382	,	385	,	387	,	390	,	393	,	396	,	398	,	401	,	404	,	407	,	410	,	413	,	416	,	419	,	421	,	424	,	427	,	430	,	433	,	436	,	439	,	442	,	445	,	448	,	451	,	454	,	457	,	461	,	464	,	467	,	470	,	473	,	476	,	479	,	483	};
  

//const int CSCTFPtLUT::dEtaCut_Low[24]    = {2,2,2,4,2,1,2,4,7,7,3,4,1,1,1,1,7,7,2,2,7,7,1,1};
//const int CSCTFPtLUT::dEtaCut_Mid[24]    = {2,2,3,5,2,2,3,5,7,7,4,5,2,2,2,2,7,7,2,2,7,7,2,2};
//const int CSCTFPtLUT::dEtaCut_High_A[24] = {3,3,4,6,3,2,4,6,7,7,5,6,2,2,2,2,7,7,3,3,7,7,2,2};
//const int CSCTFPtLUT::dEtaCut_High_B[24] = {3,3,4,7,3,3,5,7,7,7,6,7,2,2,3,3,7,7,3,3,7,7,3,2};
//const int CSCTFPtLUT::dEtaCut_High_C[24] = {4,4,5,7,4,3,6,7,7,7,7,7,3,3,3,3,7,7,4,4,7,7,3,3};
const int CSCTFPtLUT::dEtaCut_Low[24]    = {2,2,2,7,2,1,2,7,3,3,3,7,1,1,1,1,2,2,2,2,1,1,1,1};
const int CSCTFPtLUT::dEtaCut_Mid[24]    = {2,2,3,7,2,2,3,7,4,4,4,7,2,2,2,2,2,2,2,2,2,2,2,2};
const int CSCTFPtLUT::dEtaCut_High_A[24] = {3,3,4,7,3,2,4,7,5,5,5,7,2,2,2,2,3,3,3,3,2,2,2,2};
const int CSCTFPtLUT::dEtaCut_High_B[24] = {3,3,4,7,3,3,5,7,6,6,6,7,2,2,3,3,3,3,3,3,3,3,3,2};
const int CSCTFPtLUT::dEtaCut_High_C[24] = {4,4,5,7,4,3,6,7,7,7,7,7,3,3,3,3,4,4,4,4,3,3,3,3};
const int CSCTFPtLUT::dEtaCut_Open[24]   = {7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7};

const int CSCTFPtLUT::getPtbyMLH = 0xFFFF; // all modes on


CSCTFPtLUT::CSCTFPtLUT(const edm::EventSetup& es) 
    : read_pt_lut_es(true),
      read_pt_lut_file(false),
      isBinary(false)
{
	pt_method = 32;

	lowQualityFlag = 4;
	isBeamStartConf = true;

	edm::ESHandle<L1MuCSCPtLut> ptLUT;
	es.get<L1MuCSCPtLutRcd>().get(ptLUT);
        theL1MuCSCPtLut_ = ptLUT.product();

        //std::cout << "theL1MuCSCPtLut_ pointer is "
        //          << theL1MuCSCPtLut_
        //          << std::endl;

	edm::ESHandle< L1MuTriggerScales > scales ;
	es.get< L1MuTriggerScalesRcd >().get( scales ) ;
	trigger_scale = scales.product() ;

	edm::ESHandle< L1MuTriggerPtScale > ptScale ;
	es.get< L1MuTriggerPtScaleRcd >().get( ptScale ) ;
	trigger_ptscale = ptScale.product() ;

	ptMethods = CSCTFPtMethods( ptScale.product() ) ;
 
}


CSCTFPtLUT::CSCTFPtLUT(const edm::ParameterSet& pset,
		       const L1MuTriggerScales* scales,
		       const L1MuTriggerPtScale* ptScale)
  : trigger_scale( scales ),
    trigger_ptscale( ptScale ),
    ptMethods( ptScale ),
    read_pt_lut_es(false),
    read_pt_lut_file(false),
    isBinary(false)
{

  read_pt_lut_file = pset.getParameter<bool>("ReadPtLUT");
  if(read_pt_lut_file)
    {
      // if read from file, then need to set extra variables
      pt_lut_file = pset.getParameter<edm::FileInPath>("PtLUTFile");
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
  //25 and 26 like 23 and 24 correspondenly but fix high pt assignment in DT-CSC region
  // 25 - Anna's: for fw 20110118 and up, curves with MC like method 4 <- 2011 data taking
  // 26 - Anna's: for fw 20110118 and up, curves with MC like method 4 <- 2011 data taking
          //with improvments at ME1/1a: find max pt for 3 links hypothesis
  // change Quality: Q = 3 for mode 5, Quility = 2 for mode = 8, 9, 10 at eta = 1.6-1.8   
  // 27 - Anna's: for fw 20110118 and up, curves with MC like method 4 <- 2011 data taking
  // 28 - Anna's: for fw 20110118 and up, curves with MC like method 4 <- 2011 data taking
          //with improvments at ME1/1a: find max pt for 3 links hypothesis
  // 29 - Bobby's medium Quality: using fw 2012_01_31. Switch to Global Log(L). Non-Linear dphi binning. 
  // 33 - Bobby's medium Quality: using fw 2012_01_31. Switch to Global Log(L). Non-Linear dphi binning. No max pt at eta > 2.1 
  // 30 - Bobby's loose Quality: using fw 2012_01_31. Switch to Global Log(L). Non-Linear dphi binning. 
  // 31 - Bobby's tight Quality: using fw 2012_01_31. Switch to Global Log(L). Non-Linear dphi binning. 
  // 32 - Bobby's medium Quality+ {tight only mode5 at eta > 2.1}: using fw 2012_01_31. Switch to Global Log(L). Non-Linear dphi binning. 
  pt_method = pset.getUntrackedParameter<unsigned>("PtMethod",32);
  //std::cout << "pt_method from pset " << std::endl; 
  // what does this mean???
  lowQualityFlag = pset.getUntrackedParameter<unsigned>("LowQualityFlag",4);

  if(read_pt_lut_file)
    {
      pt_lut = new ptdat[1<<21];
      readLUT();
    }

  isBeamStartConf = pset.getUntrackedParameter<bool>("isBeamStartConf", true);
  
}

ptdat CSCTFPtLUT::Pt(const ptadd& address) const
{
  ptdat result;
  
  if(read_pt_lut_es) 
  {
    unsigned int shortAdd = (address.toint()& 0x1fffff);

    ptdat tmp( theL1MuCSCPtLut_->pt(shortAdd) );
  
    result = tmp;
  } 
  
  else if (read_pt_lut_file)
    {
      int shortAdd = (address.toint()& 0x1fffff);
      result = pt_lut[shortAdd];
    } 
  
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


ptdat CSCTFPtLUT::calcPt(const ptadd& address) const
{
  ptdat result;

  double Pi  = acos(-1.);
  float etaR = 0, ptR_front = 0, ptR_rear = 0, dphi12R = 0, dphi23R = 0;
  int charge12, charge23;
  unsigned type, mode, eta, fr, quality, charge, absPhi12, absPhi23;

  mode = address.track_mode;
   
  int usedetaCUT = true;
  // Chose Eta cut tightness. 1=tightest, 2=moderate, 3=loose, 4=very loose, 5=extremely loose, 6=open

  // modes 6, 7, 13
  int EtaCutLevel_1 = 2;
  int dEtaCut_1[24];
  
  for (int i=0;i<24;i++)
    {
      dEtaCut_1[i] = 10;
      if (EtaCutLevel_1 == 1)
        dEtaCut_1[i] = dEtaCut_Low[i];
      else if (EtaCutLevel_1 == 2)
        dEtaCut_1[i] = dEtaCut_Mid[i];
      else if (EtaCutLevel_1 == 3)
        dEtaCut_1[i] = dEtaCut_High_A[i];
      else if (EtaCutLevel_1 == 4)
        dEtaCut_1[i] = dEtaCut_High_B[i];
      else if (EtaCutLevel_1 == 5)
        dEtaCut_1[i] = dEtaCut_High_C[i];
      else if (EtaCutLevel_1 == 6)
        dEtaCut_1[i] = dEtaCut_Open[i];
    }
  // modes 8, 9, 10
  int EtaCutLevel_2 = 2;
  int dEtaCut_2[24];
  
  for (int i=0;i<24;i++)
    {
      dEtaCut_2[i] = 10;
      if (EtaCutLevel_2 == 1)
        dEtaCut_2[i] = dEtaCut_Low[i];
      else if (EtaCutLevel_2 == 2)
        dEtaCut_2[i] = dEtaCut_Mid[i];
      else if (EtaCutLevel_2 == 3)
        dEtaCut_2[i] = dEtaCut_High_A[i];
      else if (EtaCutLevel_2 == 4)
        dEtaCut_2[i] = dEtaCut_High_B[i];
      else if (EtaCutLevel_2 == 5)
        dEtaCut_2[i] = dEtaCut_High_C[i];
      else if (EtaCutLevel_2 == 6)
        dEtaCut_2[i] = dEtaCut_Open[i];

      float scalef = 1.0;
      if (mode == 8 || mode == 10)
        dEtaCut_2[i] = scalef*dEtaCut_2[i];
      
    }
  
  

  eta = address.track_eta;
 
  fr = address.track_fr;
  charge = address.delta_phi_sign;
  quality = trackQuality(eta, mode, fr);
  unsigned front_pt, rear_pt;
  front_pt = 0.; rear_pt = 0.;
  unsigned front_quality, rear_quality;

  etaR = trigger_scale->getRegionalEtaScale(2)->getLowEdge(2*eta+1);

  front_quality = rear_quality = quality;

  unsigned int remerged;
  int iME11;
  int CLCT_pattern;
  int dEta;
  int index = 0;
  float bestLH = -999;
 float bestLH_front = -999.0;
 float bestLH_rear = -999.0;

 int PtbyMLH = false;
  
  //***************************************************//
  if(pt_method >= 29 && pt_method <= 33)
    {
        // using fw 2012_01_31. Switch to Global Log(L). Non-Linear dphi binning.
      PtbyMLH = 0x1 & (getPtbyMLH >> (int)mode);
      ///////////////////////////////////////////////////////////
      // switch off any improvment for eta > 2.1
      if(etaR > 2.1){
         usedetaCUT = false;
         PtbyMLH = 0x0;
      }
      ///////////////////////////////////////////////////////////
      
      switch(mode)
        {
        case 2:
        case 3:
        case 4:
        case 5:
  
      
      charge12 = 1;

      // First remake the 12-bit dPhi word from the core
      remerged = (address.delta_phi_12 | (address.delta_phi_23 << 8 ) );
      
      // Now separate it into 7-bit dPhi12 and 5-bit dPhi23 parts
      absPhi12 = ((1<<7)-1) &  remerged; 
      absPhi23 = ((1<<5)-1) & (remerged >> 7);

      // Now get the corresponding dPhi value in our phi-units using the inverse dPhi LUTs
      absPhi12 = dPhiNLBMap_7bit[absPhi12];
      absPhi23 = dPhiNLBMap_5bit[absPhi23];

      if(charge) charge23 = 1;
      else charge23 = -1;

      dphi12R = (static_cast<float>(absPhi12)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
      dphi23R = (static_cast<float>(absPhi23)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
      if(charge12 * charge23 < 0) dphi23R = -dphi23R;

      ptR_front = ptMethods.Pt3Stn2012(int(mode), etaR, dphi12R, dphi23R, PtbyMLH, bestLH, 1, int(pt_method));
      bestLH_front = bestLH;
      ptR_rear  = ptMethods.Pt3Stn2012(int(mode), etaR, dphi12R, dphi23R, PtbyMLH, bestLH, 0, int(pt_method));    
      bestLH_rear = bestLH;
      
      if((pt_method == 29 || pt_method == 32 || pt_method == 30 || pt_method == 31) && mode != 5 && etaR > 2.1)//exclude mode without ME11a
        {
            float dphi12Rmin = dphi12R - Pi*10/180/3; // 10/3 degrees 
            float dphi12Rmax = dphi12R + Pi*10/180/3; // 10/3 degrees
            float dphi23Rmin = dphi23R;
            float dphi23Rmax = dphi23R;
            //if(dphi12Rmin*dphi12R < 0) dphi23Rmin = -dphi23R;
            //if(dphi12Rmax*dphi12R < 0) dphi23Rmax = -dphi23R;
            float ptR_front_min = ptMethods.Pt3Stn2012(int(mode), etaR, dphi12Rmin, dphi23Rmin, PtbyMLH, bestLH, 1, int(pt_method));
            float bestLH_front_min = bestLH;
            float ptR_rear_min = ptMethods.Pt3Stn2012(int(mode), etaR, dphi12Rmin, dphi23Rmin,  PtbyMLH, bestLH, 0, int(pt_method));
            float bestLH_rear_min = bestLH;
            float ptR_front_max = ptMethods.Pt3Stn2012(int(mode), etaR, dphi12Rmax, dphi23Rmax, PtbyMLH, bestLH, 1, int(pt_method));
            float bestLH_front_max = bestLH;
            float ptR_rear_max = ptMethods.Pt3Stn2012(int(mode), etaR, dphi12Rmax, dphi23Rmax,  PtbyMLH, bestLH, 0, int(pt_method));
            float bestLH_rear_max = bestLH;

            if (PtbyMLH)
              {
                float best_pt_front = ptR_front;
                float best_LH_front = bestLH_front;
                if (bestLH_front_min > best_LH_front)
                  {
                    best_pt_front = ptR_front_min;
                    best_LH_front = bestLH_front_min;
                  }
                if (bestLH_front_max > best_LH_front)
                  {
                    best_pt_front = ptR_front_max;
                    best_LH_front = bestLH_front_max;
                  }
                ptR_front = best_pt_front;

                float best_pt_rear = ptR_rear;
                float best_LH_rear = bestLH_rear;
                if (bestLH_rear_min > best_LH_rear)
                  {
                    best_pt_rear = ptR_rear_min;
                    best_LH_rear = bestLH_rear_min;
                  }
                if (bestLH_rear_max > best_LH_rear)
                  {
                    best_pt_rear = ptR_rear_max;
                    best_LH_rear = bestLH_rear_max;
                  }
                ptR_rear = best_pt_rear;
              }
            else
              {
                // select max pt solution for 3 links:
                ptR_front = std::max(ptR_front, ptR_front_min);
                ptR_front = std::max(ptR_front, ptR_front_max);
                ptR_rear = std::max(ptR_rear, ptR_rear_min);
                ptR_rear = std::max(ptR_rear, ptR_rear_max);
              }
        }
      break;
    case 6: // for mode 6, 7 and 13 add CLCT information in dph23 bit and iME11 in charge bit   
    case 7:
    case 13: // ME1-ME4
      
      // First remake the 12-bit dPhi word from the core
      remerged = (address.delta_phi_12 | (address.delta_phi_23 << 8 ) );
      // Now get 8-bit dPhi12 
      absPhi12 = ((1<<8)-1) & remerged; 
      // Now get 3-bit dEta
      dEta = ((1<<3)-1) & (remerged >> 8);
      // New get CLCT bit. CLCT = true if CLCTPattern = 8, 9, or 10, else 0.
      CLCT_pattern = 0x1 & (remerged >> 11);
      
      iME11 = int(charge); // = 0 if ME1/1, = 1 if ME1/2 or ME1/3 station 
      if(iME11 == 1 && etaR > 1.6) etaR = 1.55; // shift for ME1/2 station  
      if(iME11 == 0 && etaR < 1.6) etaR = 1.65; // shift for ME1/1 station

      // Get the 8-bit dPhi bin number  
      absPhi12 = ((1<<8)-1) & address.delta_phi_12;
      
      // Now get the corresponding dPhi value in our phi-units using the inverse dPhi LUTs
      absPhi12 = dPhiNLBMap_8bit[absPhi12];
      
      //int CLCT_pattern = static_cast<int>(address.delta_phi_23);
    
      dphi12R = (static_cast<float>(absPhi12)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
      
      //std::cout<< " Sector_rad = " << (CSCTFConstants::SECTOR_RAD) << std::endl;
      ptR_front = ptMethods.Pt2Stn2012(int(mode), etaR, dphi12R, PtbyMLH, bestLH, 1, int(pt_method));
      bestLH_front = bestLH;
      ptR_rear  = ptMethods.Pt2Stn2012(int(mode), etaR, dphi12R, PtbyMLH, bestLH, 0, int(pt_method));
      bestLH_rear = bestLH;
      if((pt_method == 29 || pt_method == 32 || pt_method == 30 || pt_method == 31) && etaR > 2.1)//exclude tracks without ME11a 
        {
          float dphi12Rmin = fabs(fabs(dphi12R) - Pi*10/180/3); // 10/3 degrees 
          float ptR_front_min = ptMethods.Pt2Stn2012(int(mode), etaR, dphi12Rmin,  PtbyMLH, bestLH, 1, int(pt_method));
          float bestLH_front_min = bestLH;
          float ptR_rear_min = ptMethods.Pt2Stn2012(int(mode), etaR, dphi12Rmin,   PtbyMLH, bestLH, 0, int(pt_method));
          float bestLH_rear_min = bestLH;

          if (PtbyMLH)
            {
              ptR_front = bestLH_front > bestLH_front_min ? ptR_front : ptR_front_min;
              ptR_rear  = bestLH_rear  > bestLH_rear_min ? ptR_rear : ptR_rear_min;
            }
          else
            {
              // select max pt solution for 3 links:
              ptR_front = std::max(ptR_front, ptR_front_min);
              ptR_rear = std::max(ptR_rear, ptR_rear_min);
            }
        }
      
      if( (!CLCT_pattern) && (ptR_front > 5.)) ptR_front = 5.;
      if( (!CLCT_pattern) && (ptR_rear > 5.)) ptR_rear = 5.;

       // Check dEta against reasonable values for high-pt muons
      index = 0;
      if (mode == 6) index = 0;
      if (mode == 7) index = 4;
      if (mode == 13) index = 8;
      
      if (usedetaCUT)
        {
          if (fabs(etaR)>1.2 && fabs(etaR)<=1.5)
            if (dEta>dEtaCut_1[index+0] )
              {
                if (ptR_front > 5) ptR_front = 5;
                if (ptR_rear  > 5) ptR_rear  = 5;
              }
          if (fabs(etaR)>1.5 && fabs(etaR)<=1.65)
            if (dEta>dEtaCut_1[index+1])
              {
                if (ptR_front > 5) ptR_front = 5;
                if (ptR_rear  > 5) ptR_rear  = 5;
              }
          
          if (fabs(etaR)>1.65 && fabs(etaR)<=2.1)
            if (dEta>dEtaCut_1[index+2] )
              {
                if (ptR_front > 5) ptR_front = 5;
                if (ptR_rear  > 5) ptR_rear  = 5;
              }
          if (fabs(etaR)>2.1)
            if (dEta>dEtaCut_1[index+3] )
              {
                if (ptR_front > 5) ptR_front = 5;
                if (ptR_rear  > 5) ptR_rear  = 5;
              }
        }
            
      break;
    
    case 8:
    case 9:
    case 10:
      
      
      // First remake the 12-bit dPhi word from the core
      remerged = (address.delta_phi_12 | (address.delta_phi_23 << 8 ) );
      // Now get 9-bit dPhi12 
      absPhi12 = ((1<<9)-1) & remerged; 
      // Now get 3-bit dEta
      dEta = ((1<<3)-1) & (remerged >> 9);
     
      dphi12R = (static_cast<float>(absPhi12)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;

      //std::cout<< " Sector_rad = " << (CSCTFConstants::SECTOR_RAD) << std::endl;
      ptR_front = ptMethods.Pt2Stn2012(int(mode), etaR, dphi12R,  PtbyMLH, bestLH, 1, int(pt_method));
      ptR_rear  = ptMethods.Pt2Stn2012(int(mode), etaR, dphi12R,  PtbyMLH, bestLH, 0, int(pt_method));

      index = 0;
      if (mode == 8) index = 12;
      if (mode == 9) index = 16;
      if (mode == 10) index = 20;

      
      
      
      if (usedetaCUT)
        {
          if (fabs(etaR)>1.2 && fabs(etaR)<=1.5)
            if (dEta>dEtaCut_2[index+0] )
              {
                if (ptR_front > 5) ptR_front = 5;
                if (ptR_rear  > 5) ptR_rear  = 5;
              }
          if (fabs(etaR)>1.5 && fabs(etaR)<=1.65)
            if (dEta>dEtaCut_2[index+1])
              {
                if (ptR_front > 5) ptR_front = 5;
                if (ptR_rear  > 5) ptR_rear  = 5;
              }
          
          if (fabs(etaR)>1.65 && fabs(etaR)<=2.1)
            if (dEta>dEtaCut_2[index+2] )
              {
                if (ptR_front > 5) ptR_front = 5;
                if (ptR_rear  > 5) ptR_rear  = 5;
              }
          if (fabs(etaR)>2.1)
            if (dEta>dEtaCut_2[index+3] )
              {
                if (ptR_front > 5) ptR_front = 5;
                if (ptR_rear  > 5) ptR_rear  = 5;
              }
        }
            
      break;
    // for overlap DT-CSC region using curves from data 2010
    case 11: // FR = 1 -> b1-1-3,     FR = 0 -> b1-3 
    case 12: // FR = 1 -> b1-2-3,     FR = 0 -> b1-2 
    case 14: // FR = 1 -> b1-1-2-(3), FR = 0 -> b1-1
    
      
    //sign definition: sign dphi12 = Phi_DT - Phi_CSC
    //                 sing dphi23 = 5th sign. bit of phiBend
    // -> charge = 1 -> dphi12 = +, phiBend = -
    // -> charge = 0 -> dphi12 = +, phiBend = +    
        charge12 = 1;

        // DT tracks are still using linear dPhi binning
        absPhi12 = address.delta_phi_12;
        absPhi23 = address.delta_phi_23;

        if(charge) charge23 = -1;
        else charge23 = 1;

        dphi12R = (static_cast<float>(absPhi12<<1)) / (static_cast<float>(1<<12)) * CSCTFConstants::SECTOR_RAD;
        dphi23R = float(absPhi23);
        if(charge12 * charge23 < 0) dphi23R = -dphi23R;

        int mode1;
        mode1 = int(mode);
        if(fr == 1 && mode1 == 11) mode1 = 14; // 3 station track we use dphi12 and phiBend for 2 and 3 station track

        ptR_front = ptMethods.Pt3Stn2012_DT(mode1, etaR, dphi12R, dphi23R,  PtbyMLH, bestLH, int(0), int(pt_method));
        ptR_rear = ptMethods.Pt3Stn2012_DT(mode1, etaR, dphi12R, dphi23R,   PtbyMLH, bestLH, int(0), int(pt_method));

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

  } //end pt_methods 29


//***************************************************//
  if(pt_method >= 23 && pt_method <= 28){ //here we have only pt_methods greater then 
                       //for fw 20110118 <- 2011 data taking, curves from MC like method 4
  // mode definition you could find at page 6 & 7: 
  // http://www.phys.ufl.edu/~madorsky/sp/2011-11-18/sp_core_interface.pdf 
  // it is valid starting the beggining of 2011 
  //std::cout << " pt_method = " << pt_method << std::endl;//test 
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

      if((pt_method == 24 || pt_method == 26 || pt_method == 28) && mode != 5 && etaR > 2.1)//exclude mode without ME11a
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
      if((pt_method == 24 || pt_method == 26 || pt_method == 28) && etaR > 2.1)//exclude tracks without ME11a 
        {
           float dphi12Rmin = fabs(fabs(dphi12R) - Pi*10/180/3); // 10/3 degrees 
           float ptR_front_min = ptMethods.Pt2Stn2010(int(mode), etaR, dphi12Rmin, 1, int(pt_method));
           float ptR_rear_min = ptMethods.Pt2Stn2010(int(mode), etaR, dphi12Rmin, 0, int(pt_method));
           // select max pt solution for 3 links:
           ptR_front = std::max(ptR_front, ptR_front_min);
           ptR_rear = std::max(ptR_rear, ptR_rear_min);
        }
      if( ((CLCT_pattern < 8) || (CLCT_pattern > 10)) && (ptR_front > 5.)) ptR_front = 5.;
      if( ((CLCT_pattern < 8) || (CLCT_pattern > 10)) && (ptR_rear > 5.)) ptR_rear = 5.;
      //std::cout << "mode = "<< mode << " CLCT_pattern = " << CLCT_pattern << " ptR_rear = " << ptR_rear << std::endl;

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

    //sign definition: sign dphi12 = Phi_DT - Phi_CSC
    //                 sing dphi23 = 5th sign. bit of phiBend
    // -> charge = 1 -> dphi12 = +, phiBend = -
    // -> charge = 0 -> dphi12 = +, phiBend = +    
        charge12 = 1;
        absPhi12 = address.delta_phi_12;
        absPhi23 = address.delta_phi_23;

        if(charge) charge23 = -1;
        else charge23 = 1;

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

  } //end pt_methods 23 - 28 

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
      if( ((CLCT_pattern < 8) || (CLCT_pattern > 10)) && (ptR_front > 5.)) ptR_front = 5.;
      if( ((CLCT_pattern < 8) || (CLCT_pattern > 10)) && (ptR_rear > 5.)) ptR_rear = 5.;

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

    //sign definition: sign dphi12 = Phi_DT - Phi_CSC
    //                 sing dphi23 = 5th sign. bit of phiBend
    // -> charge = 1 -> dphi12 = +, phiBend = -
    // -> charge = 0 -> dphi12 = +, phiBend = +    
        charge12 = 1;
        absPhi12 = address.delta_phi_12;
        absPhi23 = address.delta_phi_23;

        if(charge) charge23 = -1;
        else charge23 = 1;

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
      if(pt_method == 32 && eta >= 12) quality = 2; // eta > 2.1  
      break;
    case 3:
    case 4:
      /// DEA try increasing quality
      //        quality = 2;
      quality = 3;
      if(pt_method == 32 && eta >= 12) quality = 2; // eta > 2.1  
      break;
    case 5:
      quality = 1;
      if (isBeamStartConf && eta >= 12 && pt_method < 20) // eta > 2.1
	quality = 3;
      if(pt_method == 27 || pt_method == 28 || pt_method == 29 || pt_method == 32 || pt_method == 30 || pt_method == 33) quality = 3;// all mode = 5 set to quality 3 due to a lot dead ME1/1a stations
      break;
    case 6:
      if (eta>=3) // eta > 1.2
	quality = 2;
      else
	quality = 1;
      if(pt_method == 32 && eta >= 12) quality = 1; // eta > 2.1  
      break;
    case 7:
      quality = 2;
      if(pt_method > 10 && eta < 3) quality = 1; //eta < 1.2  
      if(pt_method == 32 && eta >= 12) quality = 1; // eta > 2.1  
      break;
    case 8:
    case 9:
    case 10:
      quality = 1;
      if (isBeamStartConf && eta >= 12 && pt_method < 20) // eta > 2.1
	quality = 2;
      if((pt_method == 27 || pt_method == 28 || pt_method == 30) && (eta >= 7 && eta < 9)) quality = 2; //set to quality 2 for eta = 1.6-1.8 due to a lot dead ME1/1a stations
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
      if(pt_method == 32 && eta >= 12) quality = 1; // eta > 2.1  
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


