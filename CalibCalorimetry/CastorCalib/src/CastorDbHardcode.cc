//
// F.Ratnikov (UMd), Dec 14, 2005
// $Id: CastorDbHardcode.cc,v 1.5 2012/01/12 14:15:59 muzaffar Exp $
// Adapted for Castor by L. Mundim
//
#include <vector>
#include <string>

#include "CLHEP/Random/RandGauss.h"
#include "CondFormats/CastorObjects/interface/CastorElectronicsMap.h"
#include "CalibCalorimetry/CastorCalib/interface/CastorDbHardcode.h"


CastorPedestal CastorDbHardcode::makePedestal (HcalGenericDetId fId, bool fSmear) {
  CastorPedestalWidth width = makePedestalWidth (fId);
  float value0 = fId.genericSubdet() == HcalGenericDetId::HcalGenForward ? 11. : 4.;  // fC
  float value [4] = {value0, value0, value0, value0};
  if (fSmear) {
    for (int i = 0; i < 4; i++) {
      value [i] = CLHEP::RandGauss::shoot (value0, width.getWidth (i) / 100.); // ignore correlations, assume 10K pedestal run 
      while (value [i] <= 0) value [i] = CLHEP::RandGauss::shoot (value0, width.getWidth (i));
    }
  }
  CastorPedestal result (fId.rawId (), 
		       value[0], value[1], value[2], value[3]
		       );
  return result;
}

CastorPedestalWidth CastorDbHardcode::makePedestalWidth (HcalGenericDetId fId) {
  float value = 0;
  /*
  if (fId.genericSubdet() == HcalGenericDetId::HcalGenBarrel || 
      fId.genericSubdet() == HcalGenericDetId::HcalGenOuter) value = 0.7;
  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenEndcap) value = 0.9;
  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenForward) value = 2.5;
  */
  // everything in fC
  CastorPedestalWidth result (fId.rawId ());
  for (int i = 0; i < 4; i++) {
    double width = value;
    for (int j = 0; j < 4; j++) {
      result.setSigma (i, j, i == j ? width * width : 0);
    }
  } 
  return result;
}

CastorGain CastorDbHardcode::makeGain (HcalGenericDetId fId, bool fSmear) {
  CastorGainWidth width = makeGainWidth (fId);
  float value0 = 0;
  if (fId.genericSubdet() != HcalGenericDetId::HcalGenForward) value0 = 0.177;  // GeV/fC
  else {
    if (HcalDetId(fId).depth() == 1) value0 = 0.2146;
    else if (HcalDetId(fId).depth() == 2) value0 = 0.3375;
  }
  float value [4] = {value0, value0, value0, value0};
  if (fSmear) for (int i = 0; i < 4; i++) value [i] = CLHEP::RandGauss::shoot (value0, width.getValue (i)); 
  CastorGain result (fId.rawId (), value[0], value[1], value[2], value[3]);
  return result;
}

CastorGainWidth CastorDbHardcode::makeGainWidth (HcalGenericDetId fId) {
  float value = 0;
  CastorGainWidth result (fId.rawId (), value, value, value, value);
  return result;
}

CastorQIECoder CastorDbHardcode::makeQIECoder (HcalGenericDetId fId) {
  CastorQIECoder result (fId.rawId ());
  float offset = 0;
  float slope = fId.genericSubdet () == HcalGenericDetId::HcalGenForward ? 0.36 : 0.92;  // ADC/fC
  for (unsigned range = 0; range < 4; range++) {
    for (unsigned capid = 0; capid < 4; capid++) {
      result.setOffset (capid, range, offset);
      result.setSlope (capid, range, slope);
    }
  }
  return result;
}

CastorCalibrationQIECoder CastorDbHardcode::makeCalibrationQIECoder (HcalGenericDetId fId) {
  CastorCalibrationQIECoder result (fId.rawId ());
  float lowEdges [32];
  for (int i = 0; i < 32; i++) lowEdges[i] = -1.5 + i*0.35;
  result.setMinCharges (lowEdges);
  return result;
}

CastorQIEShape CastorDbHardcode::makeQIEShape () {
  return CastorQIEShape ();
}

CastorRecoParam CastorDbHardcode::makeRecoParam (HcalGenericDetId fId) {
	CastorRecoParam result(fId.rawId(), 4, 2);
	return result;
}

CastorSaturationCorr CastorDbHardcode::makeSaturationCorr (HcalGenericDetId fId) {
	CastorSaturationCorr result(fId.rawId(), 1);
	return result;
}

#define EMAP_NHBHECR 9
#define EMAP_NHFCR 3
#define EMAP_NHOCR 4
#define EMAP_NFBR 8
#define EMAP_NFCH 3
#define EMAP_NHTRS 3
#define EMAP_NHSETS 4
#define EMAP_NTOPBOT 2
#define EMAP_NHTRSHO 4
#define EMAP_NHSETSHO 3

void CastorDbHardcode::makeHardcodeMap(CastorElectronicsMap& emap) {

  /* HBHE crate numbering */
  int hbhecrate[EMAP_NHBHECR]={0,1,4,5,10,11,14,15,17};
  /* HF crate numbering */
  int hfcrate[EMAP_NHFCR]={2,9,12};
  /* HO crate numbering */
  int hocrate[EMAP_NHOCR]={3,7,6,13};
  /* HBHE FED numbering of DCCs */
  int fedhbhenum[EMAP_NHBHECR][2]={{702,703},{704,705},{700,701},
				   {706,707},{716,717},{708,709},
				   {714,715},{710,711},{712,713}};
  /* HF FED numbering of DCCs */
  int fedhfnum[EMAP_NHFCR][2]={{718,719},{720,721},{722,723}};
  /* HO FED numbering of DCCs */
  int fedhonum[EMAP_NHOCR][2]={{724,725},{726,727},{728,729},{730,731}};
  /* HBHE/HF htr slot offsets for set of three htrs */
  int ihslot[EMAP_NHSETS]={2,5,13,16};
  /* HO htr slot offsets for three sets of four htrs */
  int ihslotho[EMAP_NHSETSHO][EMAP_NHTRSHO]={{2,3,4,5},{6,7,13,14},{15,16,17,18}};
  /* iphi (lower) starting index for each HBHE crate */
  int ihbhephis[EMAP_NHBHECR]={11,19,3,27,67,35,59,43,51};
  /* iphi (lower) starting index for each HF crate */
  int ihfphis[EMAP_NHFCR]={3,27,51};
  /* iphi (lower) starting index for each HO crate */
  int ihophis[EMAP_NHOCR]={71,17,35,53};
  /* ihbheetadepth - unique HBHE {eta,depth} assignments per fiber and fiber channel */
  int ihbheetadepth[EMAP_NHTRS][EMAP_NTOPBOT][EMAP_NFBR][EMAP_NFCH][2]={
    {{{{11,1},{ 7,1},{ 3,1}},  /* htr 0 (HB) -bot(+top) */
      {{ 5,1},{ 1,1},{ 9,1}},
      {{11,1},{ 7,1},{ 3,1}},
      {{ 5,1},{ 1,1},{ 9,1}},
      {{10,1},{ 6,1},{ 2,1}},
      {{ 8,1},{ 4,1},{12,1}},
      {{10,1},{ 6,1},{ 2,1}},
      {{ 8,1},{ 4,1},{12,1}}},
     {{{11,1},{ 7,1},{ 3,1}},  /* htr 0 (HB) +bot(-top) */
      {{ 5,1},{ 1,1},{ 9,1}},
      {{11,1},{ 7,1},{ 3,1}},
      {{ 5,1},{ 1,1},{ 9,1}},
      {{10,1},{ 6,1},{ 2,1}},
      {{ 8,1},{ 4,1},{12,1}},
      {{10,1},{ 6,1},{ 2,1}},
      {{ 8,1},{ 4,1},{12,1}}}},
    {{{{16,2},{15,2},{14,1}},  /* htr 1 (HBHE) -bot(+top) */
      {{15,1},{13,1},{16,1}},
      {{16,2},{15,2},{14,1}},
      {{15,1},{13,1},{16,1}},
      {{17,1},{16,3},{26,1}},
      {{18,1},{18,2},{26,2}},
      {{17,1},{16,3},{25,1}},
      {{18,1},{18,2},{25,2}}},
     {{{16,2},{15,2},{14,1}},  /* htr 1 (HBHE) +bot(-top) */
      {{15,1},{13,1},{16,1}},
      {{16,2},{15,2},{14,1}},
      {{15,1},{13,1},{16,1}},
      {{17,1},{16,3},{25,1}},
      {{18,1},{18,2},{25,2}},
      {{17,1},{16,3},{26,1}},
      {{18,1},{18,2},{26,2}}}},
    {{{{28,1},{28,2},{29,1}},  /* htr 2 (HE) -bot(+top) */
      {{28,3},{24,2},{24,1}},
      {{27,1},{27,2},{29,2}},
      {{27,3},{23,2},{23,1}},
      {{19,2},{20,1},{22,2}},
      {{19,1},{20,2},{22,1}},
      {{19,2},{20,1},{21,2}},
      {{19,1},{20,2},{21,1}}},
     {{{27,1},{27,2},{29,2}},  /* htr 2 (HE) +bot(-top) */
      {{27,3},{23,2},{23,1}},
      {{28,1},{28,2},{29,1}},
      {{28,3},{24,2},{24,1}},
      {{19,2},{20,1},{21,2}},
      {{19,1},{20,2},{21,1}},
      {{19,2},{20,1},{22,2}},
      {{19,1},{20,2},{22,1}}}}
  };
  /* ihfetadepth - unique HF {eta,depth} assignments per fiber and fiber channel */
  int ihfetadepth[EMAP_NTOPBOT][EMAP_NFBR][EMAP_NFCH][2]={
    {{{33,1},{31,1},{29,1}},  /* top */
     {{32,1},{30,1},{34,1}},
     {{33,2},{31,2},{29,2}},
     {{32,2},{30,2},{34,2}},
     {{34,2},{32,2},{30,2}},
     {{31,2},{29,2},{33,2}},
     {{34,1},{32,1},{30,1}},
     {{31,1},{29,1},{33,1}}},
    {{{41,1},{37,1},{35,1}},  /* bot */
     {{38,1},{36,1},{39,1}},
     {{41,2},{37,2},{35,2}},
     {{38,2},{36,2},{39,2}},
     {{40,2},{38,2},{36,2}},
     {{37,2},{35,2},{39,2}},
     {{40,1},{38,1},{36,1}},
     {{37,1},{35,1},{39,1}}}
  };
  /* ihoetasidephi - unique HO {eta,side,phi} assignments per fiber and fiber channel */
  int ihoetasidephi[EMAP_NHTRSHO][EMAP_NTOPBOT][EMAP_NFBR][EMAP_NFCH][3]={
    {{{{ 1,-1,0},{ 2,-1,0},{ 3,-1,0}},  /* htr 0 (HO) top */
      {{ 1,-1,1},{ 2,-1,1},{ 3,-1,1}},
      {{ 1,-1,2},{ 2,-1,2},{ 3,-1,2}},
      {{ 1,-1,3},{ 2,-1,3},{ 3,-1,3}},
      {{ 1,-1,4},{ 2,-1,4},{ 3,-1,4}},
      {{ 1,-1,5},{ 2,-1,5},{ 3,-1,5}},
      {{14, 1,0},{14, 1,1},{14, 1,2}},
      {{14, 1,3},{14, 1,4},{14, 1,5}}},
     {{{ 1, 1,0},{ 2, 1,0},{ 3, 1,0}},  /* htr 0 (HO) bot */
      {{ 1, 1,1},{ 2, 1,1},{ 3, 1,1}},
      {{ 1, 1,2},{ 2, 1,2},{ 3, 1,2}},
      {{ 1, 1,3},{ 2, 1,3},{ 3, 1,3}},
      {{ 1, 1,4},{ 2, 1,4},{ 3, 1,4}},
      {{ 1, 1,5},{ 2, 1,5},{ 3, 1,5}},
      {{15, 1,0},{15, 1,1},{15, 1,2}},
      {{15, 1,3},{15, 1,4},{15, 1,5}}}},
    {{{{ 6, 1,0},{ 6, 1,1},{ 6, 1,2}},  /* htr 1 (HO) top */
      {{ 6, 1,3},{ 6, 1,4},{ 6, 1,5}},
      {{ 7, 1,0},{ 7, 1,1},{ 7, 1,2}},
      {{ 7, 1,3},{ 7, 1,4},{ 7, 1,5}},
      {{ 8, 1,0},{ 8, 1,1},{ 8, 1,2}},
      {{ 8, 1,3},{ 8, 1,4},{ 8, 1,5}},
      {{ 9, 1,0},{ 9, 1,1},{ 9, 1,2}},
      {{ 9, 1,3},{ 9, 1,4},{ 9, 1,5}}},
     {{{10, 1,0},{10, 1,1},{10, 1,2}},  /* htr 1 (HO) bot */
      {{10, 1,3},{10, 1,4},{10, 1,5}},
      {{11, 1,0},{11, 1,1},{11, 1,2}},
      {{11, 1,3},{11, 1,4},{11, 1,5}},
      {{12, 1,0},{12, 1,1},{12, 1,2}},
      {{12, 1,3},{12, 1,4},{12, 1,5}},
      {{13, 1,0},{13, 1,1},{13, 1,2}},
      {{13, 1,3},{13, 1,4},{13, 1,5}}}},
    {{{{ 4,-1,0},{ 4,-1,1},{ 0, 0,0}},  /* htr 2 (HO) top */
      {{ 4,-1,2},{ 4,-1,3},{ 0, 0,0}},
      {{ 4,-1,4},{ 4,-1,5},{ 0, 0,0}},
      {{ 0, 0,0},{ 0, 0,0},{ 0, 0,0}},
      {{ 5,-1,0},{ 5,-1,1},{ 5,-1,2}},
      {{ 5,-1,3},{ 5,-1,4},{ 5,-1,5}},
      {{14,-1,0},{14,-1,1},{14,-1,2}},
      {{14,-1,3},{14,-1,4},{14,-1,5}}},
     {{{ 4, 1,0},{ 4, 1,1},{ 0, 0,0}},  /* htr 2 (HO) bot */
      {{ 4, 1,2},{ 4, 1,3},{ 0, 0,0}},
      {{ 4, 1,4},{ 4, 1,5},{ 0, 0,0}},
      {{ 0, 0,0},{ 0, 0,0},{ 0, 0,0}},
      {{ 5, 1,0},{ 5, 1,1},{ 5, 1,2}},
      {{ 5, 1,3},{ 5, 1,4},{ 5, 1,5}},
      {{15,-1,0},{15,-1,1},{15,-1,2}},
      {{15,-1,3},{15,-1,4},{15,-1,5}}}},
    {{{{ 6,-1,0},{ 6,-1,1},{ 6,-1,2}},  /* htr 3 (HO) top */
      {{ 6,-1,3},{ 6,-1,4},{ 6,-1,5}},
      {{ 7,-1,0},{ 7,-1,1},{ 7,-1,2}},
      {{ 7,-1,3},{ 7,-1,4},{ 7,-1,5}},
      {{ 8,-1,0},{ 8,-1,1},{ 8,-1,2}},
      {{ 8,-1,3},{ 8,-1,4},{ 8,-1,5}},
      {{ 9,-1,0},{ 9,-1,1},{ 9,-1,2}},
      {{ 9,-1,3},{ 9,-1,4},{ 9,-1,5}}},
     {{{10,-1,0},{10,-1,1},{10,-1,2}},  /* htr 3 (HO) bot */
      {{10,-1,3},{10,-1,4},{10,-1,5}},
      {{11,-1,0},{11,-1,1},{11,-1,2}},
      {{11,-1,3},{11,-1,4},{11,-1,5}},
      {{12,-1,0},{12,-1,1},{12,-1,2}},
      {{12,-1,3},{12,-1,4},{12,-1,5}},
      {{13,-1,0},{13,-1,1},{13,-1,2}},
      {{13,-1,3},{13,-1,4},{13,-1,5}}}} 
  };
  int ic,is,ih,itb,ifb,ifc,ifwtb,iphi_loc;
  int iside,ieta,iphi,idepth,icrate,ihtr,ihtr_fi,ifi_ch,ispigot,idcc,/*idcc_sl,*/ifed;
  std::string det;
  std::string fpga;
  // printf("      side       eta       phi     depth       det     crate       htr      fpga    htr_fi     fi_ch     spigo       dcc    dcc_sl     fedid\n");
  /* all HBHE crates */
  for(ic=0; ic<EMAP_NHBHECR; ic++){
    /* four sets of three htrs per crate */
    for(is=0; is<EMAP_NHSETS; is++){
      /* three htrs per set */
      for(ih=0; ih<EMAP_NHTRS; ih++){
	/* top and bottom */
	for(itb=0; itb<EMAP_NTOPBOT; itb++){
	  /* eight fibers per HTR FPGA */
	  for(ifb=0; ifb<EMAP_NFBR; ifb++){
	    /* three channels per fiber */
	    for(ifc=0; ifc<EMAP_NFCH; ifc++){
	      icrate=hbhecrate[ic];
	      iside=is<EMAP_NHSETS/2?-1:1;
	      ifwtb=(is/2+itb+1)%2;
	      ieta=ihbheetadepth[ih][ifwtb][ifb][ifc][0];
	      idepth=ihbheetadepth[ih][ifwtb][ifb][ifc][1];
	      ihtr=ihslot[is]+ih;
	      det=((ieta>16||idepth>2)?("HE"):("HB"));
	      fpga=((itb%2)==1)?("bot"):("top");
	      ihtr_fi=ifb+1;
	      ifi_ch=ifc;
	      iphi=(ieta>20)?(ihbhephis[ic]+(is%2)*4+itb*2-1)%72+1:(ihbhephis[ic]+(is%2)*4+itb*2+(ifb/2+is/2+1)%2-1)%72+1;
	      ispigot=(is%2)*6+ih*2+itb;
	      idcc=is<EMAP_NHSETS/2?1:2;
	      //idcc_sl=idcc==1?9:19;
	      ifed=fedhbhenum[ic][idcc-1];
	      /// load map
	      CastorElectronicsId elId(ifi_ch, ihtr_fi, ispigot, ifed-700);
	      elId.setHTR(icrate, ihtr, (fpga=="top")?(1):(0));
	      HcalDetId hId((det=="HB")?(HcalBarrel):(HcalEndcap),ieta*iside,iphi,idepth);
	      emap.mapEId2chId(elId,hId);
	      
	      //	      printf(" %9d %9d %9d %9d %9s %9d %9d %9s %9d %9d %9d %9d %9d %9d\n",iside,ieta,iphi,idepth,&det,icrate,ihtr,&fpga,ihtr_fi,ifi_ch,ispigot,idcc,idcc_sl,ifed);
	    }}}}}}
  /* all HF crates */
  for(ic=0; ic<EMAP_NHFCR; ic++){
    /* four sets of three htrs per crate */
    for(is=0; is<EMAP_NHSETS; is++){
      /* three htrs per set */
      for(ih=0; ih<EMAP_NHTRS; ih++){
	/* top and bottom */
	for(itb=0; itb<EMAP_NTOPBOT; itb++){
	  /* eight fibers per HTR FPGA */
	  for(ifb=0; ifb<EMAP_NFBR; ifb++){
	    /* three channels per fiber */
	    for(ifc=0; ifc<EMAP_NFCH; ifc++){
	      icrate=hfcrate[ic];
	      iside=is<EMAP_NHSETS/2?-1:1;
	      ieta=ihfetadepth[itb][ifb][ifc][0];
	      idepth=ihfetadepth[itb][ifb][ifc][1];
	      ihtr=ihslot[is]+ih;
	      det="HF";
	      fpga=((itb%2)==1)?("bot"):("top");
	      ihtr_fi=ifb+1;
	      ifi_ch=ifc;
	      iphi=(ieta>39)?(ihfphis[ic]+(is%2)*12+ih*4-3)%72+1:(ihfphis[ic]+(is%2)*12+ih*4+(ifb/4)*2-1)%72+1;
	      ispigot=(is%2)*6+ih*2+itb;
	      idcc=is<EMAP_NHSETS/2?1:2;
	      //idcc_sl=idcc==1?9:19;
	      ifed=fedhfnum[ic][idcc-1];
	      CastorElectronicsId elId(ifi_ch, ihtr_fi, ispigot, ifed-700);
	      elId.setHTR(icrate, ihtr, (fpga=="top")?(1):(0));
	      HcalDetId hId(HcalForward,ieta*iside,iphi,idepth);
	      emap.mapEId2chId(elId,hId);
	      // printf(" %9d %9d %9d %9d %9s %9d %9d %9s %9d %9d %9d %9d %9d %9d\n",iside,ieta,iphi,idepth,&det,icrate,ihtr,&fpga,ihtr_fi,ifi_ch,ispigot,idcc,idcc_sl,ifed);
	    }}}}}}
  /* all HO crates */
  for(ic=0; ic<EMAP_NHOCR; ic++){
    /* three sets of four htrs per crate */
    for(is=0; is<EMAP_NHSETSHO; is++){
      /* four htrs per set */
      for(ih=0; ih<EMAP_NHTRSHO; ih++){
	/* top and bottom */
	for(itb=0; itb<EMAP_NTOPBOT; itb++){
	  /* eight fibers per HTR FPGA */
	  for(ifb=0; ifb<EMAP_NFBR; ifb++){
	    /* three channels per fiber */
	    for(ifc=0; ifc<EMAP_NFCH; ifc++){
	      icrate=hocrate[ic];
	      idepth=1;
	      ieta=ihoetasidephi[ih][itb][ifb][ifc][0];
	      iside=ihoetasidephi[ih][itb][ifb][ifc][1];
	      iphi_loc=ihoetasidephi[ih][itb][ifb][ifc][2];
	      ihtr=ihslotho[is][ih];
	      det="HO";
	      fpga=((itb%2)==1)?("bot"):("top");
	      ihtr_fi=ifb+1;
	      ifi_ch=ifc;
	      iphi=(ihophis[ic]+is*6+iphi_loc-1)%72+1;
	      ispigot=ihtr<9?(ihtr-2)*2+itb:(ihtr-13)*2+itb;
	      idcc=ihtr<9?1:2;
	      //idcc_sl=idcc==1?9:19;
	      ifed=fedhonum[ic][idcc-1];
	      CastorElectronicsId elId(ifi_ch, ihtr_fi, ispigot, ifed-700);
	      elId.setHTR(icrate, ihtr, (fpga=="top")?(1):(0));
	      if (ieta==0) { // unmapped 
		emap.mapEId2chId(elId,DetId(HcalDetId::Undefined));
	      } else {
		HcalDetId hId(HcalOuter,ieta*iside,iphi,idepth+3); // HO is officially "depth=4"
		emap.mapEId2chId(elId,hId);
	      }
	      // printf(" %9d %9d %9d %9d %9s %9d %9d %9s %9d %9d %9d %9d %9d %9d\n",iside,ieta,iphi,idepth,&det,icrate,ihtr,&fpga,ihtr_fi,ifi_ch,ispigot,idcc,idcc_sl,ifed);
	    }}}}}}
  

  emap.sort();

}
