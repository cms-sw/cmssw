#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalOtherDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalLogicalMapGenerator.h"

#include <string>
#include <sstream>
#include <stdio.h>
#include <iostream>
#include <cstring>

using namespace std;

/***************************************/

HcalLogicalMapGenerator::HcalLogicalMapGenerator() {
  
  //adc and qie table; qie is entry 0, adc is entry 1. Constant across HB, HE, HO
  //  int iadcquiHBHE[NRMFIBR][NFCH][2];
 
  for (i = 0; i < NRMFIBR; i++){
    for (j = 0; j < NFCH; j++){
      //Intentionally relying on integer truncation here
      iadcquiHBHE[i][j][0] = i / 2 + 1;
      
      if (i % 2 == 0) iadcquiHBHE[i][j][1] = j;
      else            iadcquiHBHE[i][j][1] = NFCH + (j + 1) % 3;
    }
  }
}

HcalLogicalMapGenerator::~HcalLogicalMapGenerator() {

}

HcalLogicalMap HcalLogicalMapGenerator::createMap( const HcalTopology* topo, unsigned int mapIOV ) {

  mapIOV_ = mapIOV;

  std::vector <HBHEHFLogicalMapEntry> HBHEHFEntries;
  std::vector <HOHXLogicalMapEntry> HOHXEntries;
  std::vector <CALIBLogicalMapEntry> CALIBEntries;
  std::vector <ZDCLogicalMapEntry> ZDCEntries;
  std::vector <HTLogicalMapEntry> HTEntries;
  std::vector <uint32_t> LinearIndex2Entry;
  std::vector <uint32_t> HbHash2Entry;
  std::vector <uint32_t> HeHash2Entry;
  std::vector <uint32_t> HfHash2Entry;
  std::vector <uint32_t> HtHash2Entry;
  std::vector <uint32_t> HoHash2Entry;
  std::vector <uint32_t> HxCalibHash2Entry;
  //std::vector <uint32_t> CalibHash2Entry;
  std::vector <uint32_t> ZdcHash2Entry;

  int HbHalf = 1296;
  int HeHalf = 1296;
  //if (h2mode_) HeHalf = 4032;
  int HoHalf = 1080;
  int HfHalf = 864;
  int HtHalf = 2088;
  int ZdcHalf = 11;
  int CalibFull=216+425+8;

  uint32_t illegal_value=0;
  for (int i=0;i<=HcalElectronicsId::maxLinearIndex;i++) LinearIndex2Entry.push_back(illegal_value);
  for (int iHb=0;iHb<2*HbHalf;iHb++) HbHash2Entry.push_back(illegal_value);
  for (int iHe=0;iHe<2*HeHalf;iHe++) HeHash2Entry.push_back(illegal_value);
  for (int iHf=0;iHf<2*HfHalf;iHf++) HfHash2Entry.push_back(illegal_value);
  for (int iHt=0;iHt<2*HtHalf;iHt++) HtHash2Entry.push_back(illegal_value);
  for (int iHo=0;iHo<2*HoHalf;iHo++) HoHash2Entry.push_back(illegal_value);
  for (int iHcalib=0;iHcalib<CalibFull;iHcalib++) HxCalibHash2Entry.push_back(illegal_value);
  for (int iZdc=0;iZdc<2*ZdcHalf;iZdc++) ZdcHash2Entry.push_back(illegal_value);

  buildHBEFTMap(topo,HBHEHFEntries,HTEntries,LinearIndex2Entry,HbHash2Entry,HeHash2Entry,HfHash2Entry,HtHash2Entry);
  buildHOXMap(topo,HOHXEntries,LinearIndex2Entry,HoHash2Entry,HxCalibHash2Entry);
  buildCALIBMap(topo,CALIBEntries,LinearIndex2Entry,HxCalibHash2Entry);
  buildZDCMap(topo,ZDCEntries,LinearIndex2Entry,ZdcHash2Entry);

  return HcalLogicalMap(topo,HBHEHFEntries,HOHXEntries,CALIBEntries,ZDCEntries,HTEntries,
			LinearIndex2Entry,HbHash2Entry,HeHash2Entry,HfHash2Entry,HtHash2Entry,
			HoHash2Entry,HxCalibHash2Entry,/*CalibHash2Entry,*/ZdcHash2Entry);
} 

void HcalLogicalMapGenerator::buildHBEFTMap(const HcalTopology* topo,
					    std::vector <HBHEHFLogicalMapEntry>& HBHEHFEntries,
					    std::vector <HTLogicalMapEntry>& HTEntries,
					    std::vector <uint32_t>& LinearIndex2Entry,
					    std::vector <uint32_t>& HbHash2Entry,
					    std::vector <uint32_t>& HeHash2Entry,
					    std::vector <uint32_t>& HfHash2Entry,
					    std::vector <uint32_t>& HtHash2Entry) {

  /******************************/
  /* HBHE crate numbering */
  int hbhecrate_loc[NHBHECR]={0,1,4,5,10,11,14,15,17};
  memcpy( hbhecrate, hbhecrate_loc, sizeof(int)*NHBHECR );
  /* HBHE FED numbering of DCCs */
  int fedhbhenum_loc[NHBHECR][2]={{702,703},{704,705},{700,701},
				  {706,707},{716,717},{708,709},
				  {714,715},{710,711},{712,713}};
  memcpy( fedhbhenum, fedhbhenum_loc, sizeof(int)*NHBHECR*2 );
  /* HBHE/HF htr slot offsets for set of three htrs */
  int ihslot_loc[NHSETS]={2,5,13,16};
  memcpy( ihslot, ihslot_loc, sizeof(int)*NHSETS );
  /* iphi (lower) starting index for each HBHE crate */
  int ihbhephis_loc[NHBHECR]={11,19,3,27,67,35,59,43,51};
  memcpy( ihbhephis, ihbhephis_loc, sizeof(int)*NHBHECR );
  /* ihbheetadepth - unique HBHE {eta,depth} assignments per fiber and fiber channel */
  int ihbheetadepth_loc[NHTRS][NTOPBOT][NFBR][NFCH][2]={
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
  memcpy( ihbheetadepth, ihbheetadepth_loc, sizeof(int)*NHTRS*NTOPBOT*NFBR*NFCH*2 );
                                    
  //Aram's insert: I shall now define an array which contains the RM and the RM fiber for HB HE
  //and variables associated with this table
  int irm_rmfiHBHE_loc[NHTRS][NTOPBOT][NFBR][2]={
    {{{6,1},{7,1},{6,2},{7,2},{4,1},{5,1},{4,2},{5,2}},  // HTR 0 top
     {{6,3},{7,3},{6,4},{7,4},{4,3},{5,3},{4,4},{5,4}}}, // HTR 0 bot
    {{{2,1},{3,1},{2,2},{3,2},{2,1},{3,1},{2,2},{3,2}},  // HTR 1 top
     {{2,3},{3,3},{2,4},{3,4},{2,3},{3,3},{2,4},{3,4}}}, // HTR 1 bot
    {{{4,1},{5,1},{4,2},{5,2},{6,1},{7,1},{6,2},{7,2}},  // HTR 2 top
     {{4,3},{5,3},{4,4},{5,4},{6,3},{7,3},{6,4},{7,4}}}  // HTR 2 bot
  };
  memcpy( irm_rmfiHBHE, irm_rmfiHBHE_loc, sizeof(int)*NHTRS*NTOPBOT*NFBR*2 );
  //Pixel tables as a function of rm, rm fiber and fiber channel

  int ipixelHB_loc[NRMFIBR][NFCH][NRMSLOT] = {  //  fch = 0           fch = 1           fch = 2
    {{18, 17, 3,  2 }, {13, 3,  17, 7 }, {14, 1,  19, 6 }}, //rmfiber = 2
    {{19, 2,  18, 1 }, {15, 7,  13, 5 }, {17, 19, 1,  3 }}, //rmfiber = 3
    {{9,  4,  16, 11}, {5,  8,  12, 15}, {2,  13, 7,  18}}, //rmfiber = 4
    {{12, 11, 9,  8 }, {7,  15, 5,  13}, {16, 6,  14, 4 }}, //rmfiber = 5
    {{8,  5,  15, 12}, {4,  9,  11, 16}, {1,  14, 6,  19}}, //rmfiber = 6
    {{6,  16, 4,  14}, {3,  18, 2,  17}, {11, 12, 8,  9 }}  //rmfiber = 7
  };
  memcpy( ipixelHB, ipixelHB_loc, sizeof(int)*NRMFIBR*NFCH*NRMSLOT );
                                     
  int ipixelHE_loc[NRMFIBR][NFCH][NRMSLOT] = {  //  fch = 0           fch = 1           fch = 2
    {{12, 12, 12, 12}, {16, 7,  16, 7 }, {7,  16, 7,  16}}, //rmfiber = 2
    {{11, 11, 11, 11}, {19, 3,  19, 3 }, {3,  19, 3,  19}}, //rmfiber = 3
    {{15, 15, 6,  6 }, {2,  18, 2,  18}, {6,  6,  15, 15}}, //rmfiber = 4
    {{5,  14, 5,  14}, {14, 5,  14, 5 }, {18, 2,  18, 2 }}, //rmfiber = 5
    {{17, 1,  17, 1 }, {9,  9,  9,  9 }, {1,  17, 1,  17}}, //rmfiber = 6
    {{13, 4,  13, 4 }, {8,  8,  8,  8 }, {4,  13, 4,  13}}  //rmfiber = 7
  };
  memcpy( ipixelHE, ipixelHE_loc, sizeof(int)*NRMFIBR*NFCH*NRMSLOT );
  
  //slb and rct tables

  //HB and HE
 
  const char* S_slbin_odd_loc[] ={"A1","B0","B1","A0","A1","B0","B1","A0"};
  for (int gg = 0; gg < 8; gg++) {
    S_slbin_odd[gg] = S_slbin_odd_loc[gg];
  }
  const char* S_slbin_even_loc[]={"C1","D0","D1","C0","C1","D0","D1","C0"};
  for (int gg = 0; gg < 8; gg++) {
    S_slbin_even[gg] = S_slbin_even_loc[gg];
  }
  const char* rct_rackHBHE_loc[]={"S2E01-RH","S2E03-RH","S2E05-RH","S2E07-RH","S2E09-RH","S2E08-RL","S2E06-RL","S2E04-RL","S2E02-RL",
			    "S2E02-RH","S2E04-RH","S2E06-RH","S2E08-RH","S2E09-RL","S2E07-RL","S2E05-RL","S2E03-RL","S2E01-RL"};
  for (int gg = 0; gg < 18; gg++) {
    rct_rackHBHE[gg] = rct_rackHBHE_loc[gg];
  }

  int slb_table_loc[29] = {1,1,2,2,3,3,4,4,5,5,6,6,       // 1<=eta<=12
			   1,1,2,2,3,3,1,1,               // 13<=eta<=20
			   2,2,3,3,4,4,4,4,4};            // 21<=eta<=29
  memcpy( slb_table, slb_table_loc, sizeof(int)*29 );
  /********************/

  //Stream variable
  stringstream mystream;

  /* all HBHE crates */
  for(ic=0; ic<NHBHECR; ic++){
    /* four sets of three htrs per crate */
    for(is=0; is<NHSETS; is++){
      /* three htrs per set */
      for(ih=0; ih<NHTRS; ih++){
        /* top and bottom */
        for(itb=0; itb<NTOPBOT; itb++){
          /* eight fibers per HTR FPGA */
          for(ifb=0; ifb<NFBR; ifb++){
	    /* three channels per fiber */
  	    for(ifc=0; ifc<NFCH; ifc++){
              icrate=hbhecrate[ic];
              iside=is<NHSETS/2?-1:1;
              ifwtb=(is/2+itb+1)%2;
              ieta=ihbheetadepth[ih][ifwtb][ifb][ifc][0];
              idepth=ihbheetadepth[ih][ifwtb][ifb][ifc][1];
              ihtr=ihslot[is]+ih;
              (ieta>16||idepth>2) ? det = "HE": det = "HB";
              (itb%2)==1 ? fpga = "bot" : fpga = "top";
              ihtr_fi=ifb+1;
              ifi_ch=ifc;
              iphi=(ieta>20)?(ihbhephis[ic]+(is%2)*4+itb*2-1)%72+1:(ihbhephis[ic]+(is%2)*4+itb*2+(ifb/2+is/2+1)%2-1)%72+1;
              ispigot=(is%2)*6+ih*2+itb;
              idcc=is<NHSETS/2?1:2;
              idcc_sl=idcc==1?10:20;
              ifed=fedhbhenum[ic][idcc-1];
              //Aram's insert: rm variables, rbx, wedge
              //Careful here: per Pawel's map, the rm fiber is the first entry an the rm itself is the second.
              
              //If iside == -1, switch top and bottom. Why?
              if (iside == -1){
                S_side = '-';
                sidesign = 'M';
                irm    = irm_rmfiHBHE[ih][(itb + 1) % 2][ifb][1];
                irm_fi = irm_rmfiHBHE[ih][(itb + 1) % 2][ifb][0];
                
                //For eta >=21, the phi's cover 10 degrees rather than 5 (see HCAL TDR)
                if (ieta >= 21 && (irm == 1 || irm == 3)) iwedge = (iphi + 1 + irm + 1) / 4;
                else                            	  iwedge = (iphi + irm + 1) / 4;
                
                //Roll over the wedge
                if (iwedge > 18) iwedge -= 18;
              }
	      else{
	        S_side = '+';
                sidesign = 'P';
                irm    = irm_rmfiHBHE[ih][itb][ifb][1];
                irm_fi = irm_rmfiHBHE[ih][itb][ifb][0];
                
                //For eta >=21, the phi's cover 10 degrees rather than 5 (see HCAL TDR)
                if (ieta >= 21 && (irm == 4 || irm == 2)) iwedge = (iphi + 1 - irm + 6) / 4;
                else		                          iwedge = (iphi - irm + 6) / 4;
                
                //Roll over the wedge
                if (iwedge > 18) iwedge -= 18;
	      }
	      
              sprintf (tempbuff, "%s%c%2.2i%c", det.c_str(), sidesign, iwedge,'\0');
	      mystream<<tempbuff;
	      rbx = mystream.str();
	      mystream.str("");

              //Note that irm_fi ranges from 2 to 7 whereas arrays start at 0 so 
	      //I use irm_fi - 2. Likewise, irm goes from 1 to 4 so use irm - 1
	      
	      //Pixel is split by HB and HE
	      if (ieta > 16 || idepth > 2) ipixel = ipixelHE[irm_fi - 2][ifc][irm - 1]; //HE
	      else               	   ipixel = ipixelHB[irm_fi - 2][ifc][irm - 1]; //HB
	            
	      iqie = iadcquiHBHE[irm_fi - 2][ifc][0];
	      iadc = iadcquiHBHE[irm_fi - 2][ifc][1];
 
              phideg = iphi - 3;
	      if (phideg < 0) phideg = phideg + 72;
	      phideg = (phideg / 4) * 20 + 10;
	      irctcra = (( 89 - phideg  + 720)%360)/20;
	      oddcard = irctcra % 2;
	      irctcra /= 2;
	      if (iside > 0) irctcra = irctcra + 9;
	      
	      etaslb = ((ieta - 1) / 2) * 2 + 1;
	      if (etaslb > 27) etaslb = 27;
	      
	      
	      sprintf(tempbuff,"SLB_H_%3.3d%c%2.2d%c",phideg,S_side,etaslb,'\0');
	      mystream<<tempbuff;
	      slnam = mystream.str();
	      mystream.str("");

	      islb = slb_table[ieta - 1];
	      
	      // calculate RCT destination (that is, rctcon, rctcar and rctnam
	      if (ieta <= 24) { // these are the normal cards 0-5
		irctcar = 2 * ((ieta - 1)/8) + oddcard;
		irctcon = 2 * (((ieta - 1)/2)%4);
	      }
	      else {            // these are on the special card 6 which folds back eta on the odd card half
                irctcar = 6;
                eta2 = ieta;
                if (eta2 > 28) eta2 = 28;
                if (oddcard == 0) eta3 = eta2;
                else              eta3 = 57 - eta2;
                irctcon =  2 * (((eta3 - 1) / 2) % 4);
	      }
	      irctcon = 11 * irctcon + 1;

              sprintf(tempbuff,"%s-%1d-HD%2.2d",rct_rackHBHE[irctcra],irctcar,irctcon);
              mystream<<tempbuff;
              rctnam = mystream.str();
              mystream.str("");

              //Finally, the slbin
	      
              phimod8 = iphi % 8;

              for (i = 0; i < 18; i++) {
                if (iphi < i * 4 + 3) {
                  crazy = i % 2;
                  break;
               	}
              } 
	      
              int ietamod;   // determine if eta is "odd" or "even". 
              if (ieta == 29) ietamod = 0;
              else            ietamod = ieta % 2;
              if (ieta < 25) {         // use the regular table
                if (ietamod == 1) mystream<<S_slbin_odd[phimod8];
                else              mystream<<S_slbin_even[phimod8];
              }
              else if (crazy == 0) {   // use the regular table
                if (ietamod == 1) mystream<<S_slbin_odd[phimod8];
                else              mystream<<S_slbin_even[phimod8];
              }
              else {                   // swap odd/even!!!
                if (ietamod == 1) mystream<<S_slbin_even[phimod8];
                else              mystream<<S_slbin_odd[phimod8];
              }  
	      
              slbin = mystream.str();
              mystream.str(""); 

              if (ieta > 20){
                idphi = 2;
                slbin2 = slbin;
                slbin2[1] = '1';
              }
              else{
                idphi = 1;
                slbin2 = "NA";
              }

              HBHEHFLogicalMapEntry hbeflmapentry( 
						  ifi_ch, ihtr_fi, ispigot, ifed, icrate, ihtr, fpga,
						  det, iside, ieta, iphi, idepth,
						  idphi, iwedge, irm, irm_fi, ipixel, iqie, iadc,
						  islb, irctcra, irctcar, irctcon,
						  rbx, slbin, slbin2, slnam, rctnam
						  );
	      HBHEHFEntries.push_back(hbeflmapentry);
	      LinearIndex2Entry.at(hbeflmapentry.getLinearIndex())=HcalLogicalMap::makeEntryNumber(1,0,HBHEHFEntries.size()-1);

	      const HcalGenericDetId hgdi(hbeflmapentry.getDetId());
	      unsigned int denseId;
	      if (hgdi.genericSubdet()==HcalGenericDetId::HcalGenBarrel) {
		denseId=topo->detId2denseIdHB(hgdi);
		HbHash2Entry.at(denseId)=HBHEHFEntries.size();
	      }
	      if (hgdi.genericSubdet()==HcalGenericDetId::HcalGenEndcap) {
		denseId=topo->detId2denseIdHE(hgdi);
		HeHash2Entry.at(denseId)=HBHEHFEntries.size();
	      }
	      if (hgdi.genericSubdet()==HcalGenericDetId::HcalGenForward) {
		denseId=topo->detId2denseIdHF(hgdi);
		HfHash2Entry.at(denseId)=HBHEHFEntries.size();
	      }

	      ConstructTriggerTower(topo,
				    HTEntries,
				    iside, ieta, iphi, idphi, idepth, det, iwedge, irm,
				    ipixel, iqie, iadc, irm_fi, ifi_ch, icrate, ihtr, fpga,
				    ihtr_fi, ispigot, islb, slbin, slbin2,
				    slnam, irctcra, irctcar, irctcon, rctnam, ifed
				    );
            }
          }
        }
      }
    }
  }

  /********************/
  /* HF crate numbering */
  int hfcrate_loc[NHFCR]={2,9,12};
  memcpy( hfcrate, hfcrate_loc, sizeof(int)*NHFCR );
  /* HF FED numbering of DCCs */
  int fedhfnum_loc[NHFCR][2]={{718,719},{720,721},{722,723}};
  memcpy( fedhfnum, fedhfnum_loc, sizeof(int)*NHFCR*2 );
  /* iphi (lower) starting index for each HF crate */
  int ihfphis_loc[NHFCR]={3,27,51};
  memcpy( ihfphis, ihfphis_loc, sizeof(int)*NHFCR );
  /* ihfetadepth - unique HF {eta,depth} assignments per fiber and fiber channel */
  int ihfetadepth_loc[NTOPBOT][NFBR][NFCH][2]={
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
  memcpy( ihfetadepth, ihfetadepth_loc, sizeof(int)*NTOPBOT*NFBR*NFCH*2 );


  int irm_rmfiHF_loc[NHTRS][NTOPBOT][NFBR][2]={
    {{{1,2},{2,2},{3,2},{4,2},{1,3},{2,3},{3,3},{4,3}},  // HTR 0 top
     {{5,2},{6,2},{7,2},{8,2},{5,3},{6,3},{7,3},{8,3}}}, // HTR 0 bot
    {{{1,1},{2,1},{3,1},{4,1},{1,2},{2,2},{3,2},{4,2}},  // HTR 1 top
     {{5,1},{6,1},{7,1},{8,1},{5,2},{6,2},{7,2},{8,2}}}, // HTR 1 bot
    {{{1,3},{2,3},{3,3},{4,3},{1,1},{2,1},{3,1},{4,1}},  // HTR 2 top
     {{5,3},{6,3},{7,3},{8,3},{5,1},{6,1},{7,1},{8,1}}}  // HTR 2 bot
  };
  memcpy( irm_rmfiHF, irm_rmfiHF_loc, sizeof(int)*NHTRS*NTOPBOT*NFBR*2 );

  //HF
  const char* S_slbin_7_loc[] ={"A0","A1","B0","B1"};
  for (int gg = 0; gg < 4; gg++) {
    S_slbin_7[gg] = S_slbin_7_loc[gg];
  }
  const char* S_slbin_3_loc[] ={"C0","C1","D0","D1"};
  for (int gg = 0; gg < 4; gg++) {
    S_slbin_3[gg] = S_slbin_3_loc[gg];
  }
  const char* rct_rackHF_loc[]={"S2E01-FH","S2E03-FH","S2E05-FH","S2E07-FH","S2E09-FH","S2E08-FL","S2E06-FL","S2E04-FL","S2E02-FL",
			  "S2E02-FH","S2E04-FH","S2E06-FH","S2E08-FH","S2E09-FL","S2E07-FL","S2E05-FL","S2E03-FL","S2E01-FL"};
  for (int gg = 0; gg < 18; gg++) {
    rct_rackHF[gg] = rct_rackHF_loc[gg];
  }
 

  /***************/
  
  /* all HF crates */
  for(ic=0; ic<NHFCR; ic++){
    /* four sets of three htrs per crate */
    for(is=0; is<NHSETS; is++){
      /* three htrs per set */
      for(ih=0; ih<NHTRS; ih++){
        /* top and bottom */
        for(itb=0; itb<NTOPBOT; itb++){
          /* eight fibers per HTR FPGA */
          for(ifb=0; ifb<NFBR; ifb++){
            /* three channels per fiber */
            for(ifc=0; ifc<NFCH; ifc++){
              icrate=hfcrate[ic];
              iside=is<NHSETS/2?-1:1;
              ieta=ihfetadepth[itb][ifb][ifc][0];
              idepth=ihfetadepth[itb][ifb][ifc][1];
              ihtr=ihslot[is]+ih;
              det = "HF";
              (itb%2)== 1 ? fpga = "bot" : fpga = "top";
              ihtr_fi=ifb+1;
              ifi_ch=ifc;
              iphi=(ieta>39)?(ihfphis[ic]+(is%2)*12+ih*4-1)%72+1:(ihfphis[ic]+(is%2)*12+ih*4+(ifb/4)*2-1)%72+1;
              ispigot=(is%2)*6+ih*2+itb;
              idcc=is<NHSETS/2?1:2;
              idcc_sl=idcc==1?10:20;
              ifed=fedhfnum[ic][idcc-1];
              
              irm_fi = irm_rmfiHF[ih][itb][ifb][0];
              
              //Don't switch in the HF. Why?
              if (iside == -1){
                S_side = '-';
                sidesign = 'M';
                
                if (ieta < 40){
                  if      (iphi == 1)     iphi = 71;
                  else if (iphi == 71)    iphi = 1;
                  else if (iphi % 4 == 1) iphi -= 2;
                  else if (iphi % 4 == 3) iphi += 2;
                  else                    edm::LogInfo( "HcalLogicalMapGenerator") <<"Even iphi in HFM"<<endl;
                }
              }
              else{
                S_side = '+';
                sidesign = 'P';
              }
              
              //RM and RBX number
              if ((iside == 1 && ieta == 40) || (iside == -1 && ieta == 41)){
                irm = ((iphi + 1) / 2) % 36 + 1;
                hfphi = ((iphi + 1) / 6) % 12 + 1; 
              }
              else{
                irm = ( iphi + 1) / 2;
                hfphi = (iphi - 1) / 6 + 1;
              }
              irm = (irm - 1) % 3 + 1;
              
              //Wedge
              if (iphi >= 71) iwedge = 1;
              else	      iwedge = (iphi + 1) / 4 + 1;
              
              //RBX
              sprintf (tempbuff, "%s%c%2.2i%c", det.c_str(), sidesign, hfphi,'\0');
              mystream<<tempbuff;
              rbx = mystream.str();
              mystream.str("");
             
              //No pixel in HF, follow Fedor's convention
              ipixel = 0;
              
              //Integer truncation again consistent with Fedor's map. 
              iqie = (irm_fi - 1) / 2 + 1;
              
              if (irm_fi % 2 != 0) iadc = ifi_ch;
              else       	   iadc = NFCH + (ifi_ch + 1) % 3;
                 
              //slb and rct variables
              //rctcrate
              phideg = iphi - 3;
              if (phideg < 0) phideg = phideg + 72;
              phideg = (phideg / 4) * 20 + 10;
              irctcra = (( 89 - phideg  + 720)%360)/40;
              if (iside > 0) irctcra = irctcra + 9;
              
              //rct card and rct connector appear to be dummy here -- again, I follow Fedor's convention
              irctcar = 99;
              irctcon = 0;
              
              etaslb = 29;
              
              sprintf(tempbuff,"SLB_H_%3.3d%c%2.2d",phideg,S_side,etaslb);
              mystream<<tempbuff;
              slnam = mystream.str();
              mystream.str("");
              
              sprintf(tempbuff,"%s-JSC-HF_IN",rct_rackHF[irctcra]);
              mystream<<tempbuff;
              rctnam = mystream.str();
              mystream.str("");
              
              islb = 6;
              
              int phibin = (iphi + 1) % 8 ;
              int etabin = (ieta - 29) / 3;
              if (etabin < 0) etabin = 0;
              if (etabin > 3) etabin = 3;
              if (phibin < 4) mystream<<S_slbin_7[etabin];
              else            mystream<<S_slbin_3[etabin];
              
              slbin = mystream.str();
              mystream.str("");
 
              slbin2 = "NA";
             
              if (ieta < 40) idphi = 2;
              else 	     idphi = 4;

              HBHEHFLogicalMapEntry hbeflmapentry( 
						  ifi_ch, ihtr_fi, ispigot, ifed, icrate, ihtr, fpga,
						  det, iside, ieta, iphi, idepth,
						  idphi, iwedge, irm, irm_fi, ipixel, iqie, iadc,
						  islb, irctcra, irctcar, irctcon,
						  rbx, slbin, slbin2, slnam, rctnam
						  );
	      HBHEHFEntries.push_back(hbeflmapentry);
	      LinearIndex2Entry.at(hbeflmapentry.getLinearIndex())=HcalLogicalMap::makeEntryNumber(1,0,HBHEHFEntries.size()-1);

	      const HcalGenericDetId hgdi(hbeflmapentry.getDetId());
	      unsigned int denseId;
	      if (hgdi.genericSubdet()==HcalGenericDetId::HcalGenBarrel) {
		denseId=topo->detId2denseIdHB(hgdi);
		HbHash2Entry.at(denseId)=HBHEHFEntries.size();
	      }
	      if (hgdi.genericSubdet()==HcalGenericDetId::HcalGenEndcap) {
		denseId=topo->detId2denseIdHE(hgdi);
		HeHash2Entry.at(denseId)=HBHEHFEntries.size();
	      }
	      if (hgdi.genericSubdet()==HcalGenericDetId::HcalGenForward) {
		denseId=topo->detId2denseIdHF(hgdi);
		HfHash2Entry.at(denseId)=HBHEHFEntries.size();
	      }

	      ConstructTriggerTower(topo,
				    HTEntries,
				    iside, ieta, iphi, idphi, idepth, det, iwedge, irm,
				    ipixel, iqie, iadc, irm_fi, ifi_ch, icrate, ihtr, fpga,
				    ihtr_fi, ispigot, islb, slbin, slbin2,
				    slnam, irctcra, irctcar, irctcon, rctnam, ifed
				    );
            }
          }
        }
      }
    }
  }
} 

void HcalLogicalMapGenerator::buildHOXMap(const HcalTopology* topo,
					  std::vector <HOHXLogicalMapEntry>& HOHXEntries,
					  std::vector <uint32_t>& LinearIndex2Entry,
					  std::vector <uint32_t>& HoHash2Entry,
					  std::vector <uint32_t>& HxCalibHash2Entry) {
  /******************************/
  /* HO crate numbering */
  int hocrate_loc[NHOCR]={3,7,6,13};
  memcpy( hocrate, hocrate_loc, sizeof(int)*NHOCR );
  /* HO FED numbering of DCCs */
  int fedhonum_loc[NHOCR][2]={{724,725},{726,727},{728,729},{730,731}};
  memcpy( fedhonum, fedhonum_loc, sizeof(int)*NHOCR*2 );
  /* HO htr slot offsets for three sets of four htrs */
  int ihslotho_loc[NHSETSHO][NHTRSHO]={{2,3,4,5},{6,7,13,14},{15,16,17,18}};
  memcpy( ihslotho, ihslotho_loc, sizeof(int)*NHSETSHO*NHTRSHO );
  /* iphi (lower) starting index for each HO crate */
  int ihophis_loc[NHOCR]={71,17,35,53};
  memcpy( ihophis, ihophis_loc, sizeof(int)*NHOCR );
  //RM for the HO as a function of eta, phi and side as implemented in complete_ho_map.txt
  //There are only 24 phi columns because after that it begins to repeat. The relevant variable is phi mod 24.
  //HX as the 16th eta entry
  int HO_RM_table_loc[24][16][2] = 
    {
      {{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{4,2},{4,2},{4,2},{4,2},{4,2},{4,2}},
      {{2,2},{2,2},{2,2},{2,2},{4,4},{4,4},{4,4},{4,4},{4,4},{4,4},{2,4},{2,4},{2,4},{2,4},{2,4},{2,4}},
      {{3,3},{3,3},{3,3},{3,3},{4,4},{4,4},{4,4},{4,4},{4,4},{4,4},{2,4},{2,4},{2,4},{2,4},{2,4},{2,4}},
      {{3,3},{3,3},{3,3},{3,3},{4,4},{4,4},{4,4},{4,4},{4,4},{4,4},{2,4},{2,4},{2,4},{2,4},{2,4},{2,4}},
      {{4,4},{4,4},{4,4},{4,4},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{1,3},{1,3},{1,3},{1,3},{1,3},{1,3}},
      {{4,4},{4,4},{4,4},{4,4},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{1,3},{1,3},{1,3},{1,3},{1,3},{1,3}},
      {{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{1,3},{1,3},{1,3},{1,3},{1,3},{1,3}},
      {{3,3},{3,3},{3,3},{3,3},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{3,1},{3,1},{3,1},{3,1},{3,1},{3,1}},
      {{2,2},{2,2},{2,2},{2,2},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{3,1},{3,1},{3,1},{3,1},{3,1},{3,1}},
      {{2,2},{2,2},{2,2},{2,2},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{3,1},{3,1},{3,1},{3,1},{3,1},{3,1}},
      {{4,4},{4,4},{4,4},{4,4},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{4,2},{4,2},{4,2},{4,2},{4,2},{4,2}},
      {{4,4},{4,4},{4,4},{4,4},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{4,2},{4,2},{4,2},{4,2},{4,2},{4,2}},
      {{3,3},{3,3},{3,3},{3,3},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{4,2},{4,2},{4,2},{4,2},{4,2},{4,2}},
      {{3,3},{3,3},{3,3},{3,3},{4,4},{4,4},{4,4},{4,4},{4,4},{4,4},{2,4},{2,4},{2,4},{2,4},{2,4},{2,4}},
      {{2,2},{2,2},{2,2},{2,2},{4,4},{4,4},{4,4},{4,4},{4,4},{4,4},{2,4},{2,4},{2,4},{2,4},{2,4},{2,4}},
      {{2,2},{2,2},{2,2},{2,2},{4,4},{4,4},{4,4},{4,4},{4,4},{4,4},{2,4},{2,4},{2,4},{2,4},{2,4},{2,4}},
      {{1,1},{1,1},{1,1},{1,1},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{1,3},{1,3},{1,3},{1,3},{1,3},{1,3}},
      {{1,1},{1,1},{1,1},{1,1},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{1,3},{1,3},{1,3},{1,3},{1,3},{1,3}},
      {{2,2},{2,2},{2,2},{2,2},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{1,3},{1,3},{1,3},{1,3},{1,3},{1,3}},
      {{2,2},{2,2},{2,2},{2,2},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{3,1},{3,1},{3,1},{3,1},{3,1},{3,1}},
      {{3,3},{3,3},{3,3},{3,3},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{3,1},{3,1},{3,1},{3,1},{3,1},{3,1}},
      {{3,3},{3,3},{3,3},{3,3},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{3,1},{3,1},{3,1},{3,1},{3,1},{3,1}},
      {{1,1},{1,1},{1,1},{1,1},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{4,2},{4,2},{4,2},{4,2},{4,2},{4,2}},
      {{1,1},{1,1},{1,1},{1,1},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{4,2},{4,2},{4,2},{4,2},{4,2},{4,2}}
    };
  memcpy( HO_RM_table, HO_RM_table_loc, sizeof(int)*24*16*2 );

  //For |eta| 5 to 15, rm_fi is a function of |eta| only while htr_fi is a function of side and |eta|
  int HO_RM_fi_eta5to15_loc[11] = {3, 2, 5, 4, 7, 6, 3, 2, 5, 4, 7};
  memcpy( HO_RM_fi_eta5to15, HO_RM_fi_eta5to15_loc, sizeof(int)*11 );
  //For eta=16 it is 6, declared in the header
  HO_RM_fi_eta16 = 6;
  //For eta in the YB0 region, rm_fi is dependent on side, 4 for -1, and 5 for +1
  //but the values won't be any different than when the regular loop is executed
  //int HO_RM_fi_etaYB0[2] = {4, 5}
  
  int HO_htr_fi_450eta5to15_loc[2][11] = {{2, 2, 4, 6, 8, 2, 4, 6, 8, 4, 6},   //iside = -1
					  {2, 8, 6, 4, 2, 8, 6, 4, 2, 4, 2}};  //iside = +1
  memcpy( HO_htr_fi_450eta5to15, HO_htr_fi_450eta5to15_loc, sizeof(int)*2*11 );
  // for the minus side, htr_fi is determined by the RM, will have to shift up one since rm is 
  //numbered 1-4 not 0-3	
  int HO_htr_fi_450eta16M_loc[4] = {7, 7, 8, 8};//iside = -1
  memcpy( HO_htr_fi_450eta16M, HO_htr_fi_450eta16M_loc, sizeof(int)*4 );
  //For the plus side, there is a new htr so all values can be used, but only 6 are used
  //Depend on phi, but also on rm and sector (2,4,6,8,10,12)
  int HO_htr_fi_450eta16P_loc[4][6] = { {4,2,4,4,2,6},{2,6,6,2,6,4},{5,3,5,5,3,7},{3,7,7,3,7,5} };                 //iside = +1
  memcpy( HO_htr_fi_450eta16P, HO_htr_fi_450eta16P_loc, sizeof(int)*4*6 );
				      
  //for |eta| <= 4, htr_fi is a function of side and phmod6
  int HO_htr_fi_eta4_loc[2][6] = {{6, 6, 8, 8, 7, 7},   //iside = -1
				  {4, 4, 5, 5, 3, 3}};  //iside = +1
  memcpy( HO_htr_fi_eta4, HO_htr_fi_eta4_loc, sizeof(int)*2*6 );

  int HO_htr_fi_eta123_loc[2][6] = {{6, 5, 4, 3, 8, 7},   //iside = -1
				    {8, 7, 6, 5, 2, 1}};  //iside = +1
  memcpy( HO_htr_fi_eta123, HO_htr_fi_eta123_loc, sizeof(int)*2*6 );
  
  //HO_htr_fi_123eta5to15[2][11] is unnecessary because HO_htr_fi_123eta5to15[i][j] == int HO_htr_fi_450eta5to15[i][j] - 1
    
  //For |eta| 1 to 4, it is a function of phi, eta and side. eta 1-3 always have the same value given a side, eta 4 is separate
  //and thus gets its own box
  int HO_RM_fi_eta1to4_loc[72][2][2] = 
    {           //side = -1            side = 1
      {{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}},{{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}}, //Phi 1  to 8
      {{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}}, //Phi 9  to 16
      {{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}}, //Phi 17 to 24
      {{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}},{{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}}, //Phi 25 to 32
      {{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}}, //Phi 33 to 40
      {{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}}, //Phi 41 to 48
      {{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}},{{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}}, //Phi 49 to 56
      {{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}},{{2,6},{5,4}},{{3,7},{5,4}}, //Phi 57 to 64
      {{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}},{{7,3},{4,5}},{{6,2},{4,5}}  //Phi 65 to 72
    };
  memcpy( HO_RM_fi_eta1to4, HO_RM_fi_eta1to4_loc, sizeof(int)*72*2*2 );

  //Pixel and letter code for the HO. Ring 0 is separate and goes into entry 0, Rings +/- 1,2 are all the same and go to entry 1.
  //Pixel and let_code for HO ring 0 on the crosstalk channels: on rm_fi 4, pixel = 18, let_code = X
  //Pixel and let_code for HO ring 0 on the crosstalk channels: on rm_fi 5, pixel = 2, let_code = X
  //                    Fiber Channel       0        1       2         0       1       2          0       1       2
  int ipixelHO_loc[NRMFIBR][NFCH][2] = {{{12,12},{ 7, 7},{ 6, 3}}, {{ 4, 4},{ 8, 8},{ 5, 1}}, {{19,11},{18, 6},{17, 2}},   //RM fibers 2,3,4
					{{ 2, 9},{ 1,13},{ 3, 5}}, {{11,19},{16,18},{15,17}}, {{13,15},{ 9,14},{14,16}}}; //RM fibers 5,6,7
  memcpy( ipixelHO, ipixelHO_loc, sizeof(int)*NRMFIBR*NFCH*2 );
  //                            Fiber Channel        0         1         2            0         1         2            0       Y 1         2
  std::string letterHO_loc[NRMFIBR][NFCH][2] = {{{"E","E"},{"G","L"},{"F","S"}}, {{"Q","M"},{"N","T"},{"P","F"}}, {{"A","C"},{"X","J"},{"J","Q"}},
						{{"X","K"},{"R","R"},{"H","D"}}, {{"D","A"},{"C","G"},{"B","N"}}, {{"L","H"},{"M","P"},{"K","B"}}}; 

  for (int jj = 0; jj < NRMFIBR; jj++) {
    for (int kk = 0; kk < NFCH; kk++) {
      for (int ll = 0; ll < 2; ll++) {
	letterHO[jj][kk][ll] = letterHO_loc[jj][kk][ll];
      }
    }
  }

  /******************************************************************************************/
  //  Here is the section that deals with the miscabled HO RBXs.
  //  To modify the code between IOVs, just note that inverted corresponds to -infinity to end of 2008
  //  and otherwise, to 2009 to infinity.
  //  Except for some small but major caveats:
  //    HO2M04 (phi=16) was not able to be corrected, thus it is still inverted
  //    HO2M06 (phi=31) has not been active in 2009, and the cables have not been switched officially
  //    HO2P12 (phi=67) has not been active in 2009, and the cables have not been switched officially
  //  In the map, the inactive RBXs have been switched since the changes will be effected when the HO SiPMs
  //  are installed, also at that time, if it is possible, HO2M04 will be corrected as well.

  //              switched HO RM's need reversed eta values
  if(mapIOV_==1) {
    int rmspecialeta_loc[6][6] = { //there are 6 special cases, corresponding to 6 values of phi
      { 14, 15, 14, 13, 12, 11 }, // ring +2 phi = 5, inverted
      { 14, 15, 14, 13, 12, 11 }, // ring +2 phi = 67, inverted
      { 10, 9, 8, 7, 6, 5 },      // ring -1 phi = 57, inverted
      { 10, 9, 8, 7, 6, 5 },      // ring -1 phi = 65, inverted
      { 14, 15, 14, 13, 12, 11 }, // ring -2 phi = 16, inverted
      { 14, 15, 14, 13, 12, 11 }  // ring -2 phi = 31, inverted
    };

    std::string rmspeciallet_code_loc[6][6] = { //there are 6 special cases, corresponding to 6 values of phi
      { "X", "B", "C", "D", "E", "F" },  // ring +2 phi = 5, inverted
      { "X", "P", "Q", "R", "S", "T" },  // ring +2 phi = 67, inverted
      { "G", "H", "J", "K", "L", "M" },  // ring -1 phi = 57, inverted
      { "A", "B", "C", "D", "E", "F" },  // ring -1 phi = 65, inverted
      { "X", "B", "C", "D", "E", "F" },  // ring -2 phi = 16, inverted
      { "X", "P", "Q", "R", "S", "T" }   // ring -2 phi = 31, inverted
    };

    std::string rmspecialdet_loc[6][6] = { //there are 6 special cases, corresponding to 6 values of phi
      { "HOX", "HO", "HO", "HO", "HO", "HO" }, // ring +2 phi = 5, inverted
      { "HOX", "HO", "HO", "HO", "HO", "HO" }, // ring +2 phi = 67, inverted
      { "HO", "HO", "HO", "HO", "HO", "HO" },  // ring -1 phi = 57, inverted
      { "HO", "HO", "HO", "HO", "HO", "HO" },  // ring -1 phi = 65, inverted
      { "HOX", "HO", "HO", "HO", "HO", "HO" }, // ring -2 phi = 16, inverted
      { "HOX", "HO", "HO", "HO", "HO", "HO" }  // ring -2 phi = 31, inverted
    };

    memcpy( rmspecialeta, rmspecialeta_loc, sizeof(int)*6*6 );
    //              switched HO RM's need revised letter codes
  
    for (int jj = 0; jj < 6; jj++) {
      for (int kk = 0; kk < 6; kk++) {
	rmspeciallet_code[jj][kk] = rmspeciallet_code_loc[jj][kk];}}
  
    for (int jj = 0; jj < 6; jj++) {
      for (int kk = 0; kk < 6; kk++) {
	rmspecialdet[jj][kk] = rmspecialdet_loc[jj][kk];}}
  }

  else if (mapIOV_==2) {
    int rmspecialeta_loc[6][6] = { //there are 6 special cases, corresponding to 6 values of phi
      { 11, 12, 13, 14, 15, 15 }, // ring +2 phi = 5
      { 11, 12, 13, 14, 15, 15 }, // ring +2 phi = 67
      { 5, 6, 7, 8, 9, 10 },      // ring -1 phi = 57
      { 5, 6, 7, 8, 9, 10 },      // ring -1 phi = 65
      { 14, 15, 14, 13, 12, 11 }, // ring -2 phi = 16, still inverted
      { 11, 12, 13, 14, 15, 15 }  // ring -2 phi = 31
    };

    std::string rmspeciallet_code_loc[6][6] = { //there are 6 special cases, corresponding to 6 values of phi
      { "F", "E", "D", "C", "B", "X" },  // ring +2 phi = 5
      { "T", "S", "R", "Q", "P", "X" },  // ring +2 phi = 67
      { "M", "L", "K", "J", "H", "G" },  // ring -1 phi = 57
      { "F", "E", "D", "C", "B", "A" },  // ring -1 phi = 65
      { "X", "B", "C", "D", "E", "F" },  // ring -2 phi = 16,still  inverted
      { "T", "S", "R", "Q", "P", "X" }   // ring -2 phi = 31
    };

    std::string rmspecialdet_loc[6][6] = { //there are 6 special cases, corresponding to 6 values of phi
      { "HO", "HO", "HO", "HO", "HO", "HOX" }, // ring +2 phi = 5
      { "HO", "HO", "HO", "HO", "HO", "HOX" }, // ring +2 phi = 67
      { "HO", "HO", "HO", "HO", "HO", "HO" },  // ring -1 phi = 57
      { "HO", "HO", "HO", "HO", "HO", "HO" },  // ring -1 phi = 65
      { "HOX", "HO", "HO", "HO", "HO", "HO" }, // ring -2 phi = 16, still inverted
      { "HO", "HO", "HO", "HO", "HO", "HOX" }  // ring -2 phi = 31
    };

    memcpy( rmspecialeta, rmspecialeta_loc, sizeof(int)*6*6 );
    //              switched HO RM's need revised letter codes
  
    for (int jj = 0; jj < 6; jj++) {
      for (int kk = 0; kk < 6; kk++) {
	rmspeciallet_code[jj][kk] = rmspeciallet_code_loc[jj][kk];}}
  
    for (int jj = 0; jj < 6; jj++) {
      for (int kk = 0; kk < 6; kk++) {
	rmspecialdet[jj][kk] = rmspecialdet_loc[jj][kk];}}
  }

  else {
    int rmspecialeta_loc[6][6] = { //there are 6 special cases, corresponding to 6 values of phi
      { 11, 12, 13, 14, 15, 15 }, // ring +2 phi = 5
      { 11, 12, 13, 14, 15, 15 }, // ring +2 phi = 67
      { 5, 6, 7, 8, 9, 10 },      // ring -1 phi = 57
      { 5, 6, 7, 8, 9, 10 },      // ring -1 phi = 65
      { 11, 12, 13, 14, 15, 15 }, // ring -2 phi = 16
      { 11, 12, 13, 14, 15, 15 }  // ring -2 phi = 31
    };

    std::string rmspeciallet_code_loc[6][6] = { //there are 6 special cases, corresponding to 6 values of phi
      { "F", "E", "D", "C", "B", "X" },  // ring +2 phi = 5
      { "T", "S", "R", "Q", "P", "X" },  // ring +2 phi = 67
      { "M", "L", "K", "J", "H", "G" },  // ring -1 phi = 57
      { "F", "E", "D", "C", "B", "A" },  // ring -1 phi = 65
      { "F", "E", "D", "C", "B", "X" },  // ring -2 phi = 16
      { "T", "S", "R", "Q", "P", "X" }   // ring -2 phi = 31
    };

    std::string rmspecialdet_loc[6][6] = { //there are 6 special cases, corresponding to 6 values of phi
      { "HO", "HO", "HO", "HO", "HO", "HOX" }, // ring +2 phi = 5
      { "HO", "HO", "HO", "HO", "HO", "HOX" }, // ring +2 phi = 67
      { "HO", "HO", "HO", "HO", "HO", "HO" },  // ring -1 phi = 57
      { "HO", "HO", "HO", "HO", "HO", "HO" },  // ring -1 phi = 65
      { "HO", "HO", "HO", "HO", "HO", "HOX" }, // ring -2 phi = 16
      { "HO", "HO", "HO", "HO", "HO", "HOX" }  // ring -2 phi = 31
    };

    memcpy( rmspecialeta, rmspecialeta_loc, sizeof(int)*6*6 );
    //              switched HO RM's need revised letter codes
  
    for (int jj = 0; jj < 6; jj++) {
      for (int kk = 0; kk < 6; kk++) {
	rmspeciallet_code[jj][kk] = rmspeciallet_code_loc[jj][kk];}}
  
    for (int jj = 0; jj < 6; jj++) {
      for (int kk = 0; kk < 6; kk++) {
	rmspecialdet[jj][kk] = rmspecialdet_loc[jj][kk];}}
  }

  /******************************/  

  //Stream variable
  stringstream mystream;

  //Radical change: HO iterates over eta and phi rather than crate, HTR, etc. 
  for(isid = -1; isid < 2; isid+=2){
    for (iph = 0; iph < NHOPHI; iph++){
      for (iet = 0; iet < NHOETA; iet++){
        
        iphi = iph + 1;
        ieta = iet + 1;
        iside = isid;
        
        if (iphi >= 71 || iphi < 17)      ic = 0;
        else if (iphi >= 17 && iphi < 35) ic = 1;
        else if (iphi >= 35 && iphi < 53) ic = 2;
        else                              ic = 3;
        	
        icrate=hocrate[ic];
        idepth=4;
        det = "HO";
        
        //fpga = top/bottom for Ring 0 depends on a pattern that repeats every 30 degrees (6 phi)
        //Hence, phmod6 (not phi mod 6 because I don't want to separate 71 and 72, etc.)
        
        phmod6 = iph % 6;
        //Bools associated with phmod6 to be used with htr_fi and the patch panel
        phmod6e450 = (phmod6 == 4 || phmod6 == 5 || phmod6 == 0); 
        phmod6e123 = (phmod6 == 1 || phmod6 == 2 || phmod6 == 3); 
        
        //Ring 0 (HTR type 0) is special
        if      (ieta <= 3  && (iside < 0 || phmod6 >= 4)) fpga = "bot";
        else if (ieta <= 3  && iside > 0 && phmod6 < 4)  fpga = "top";
        //new HX HTR
        else if ((ieta > 15 && iside > 0) && (icrate == 3 || icrate == 6))   fpga = "top";
        else if ((ieta > 15 && iside > 0) && (icrate == 7 || icrate == 13))   fpga = "bot";
        //HTR types 0 and 1
        else if (ieta >= 10 && iside > 0)                 fpga = "top";
        else if (ieta <  10 && ieta >= 6 && iside > 0)    fpga = "bot";
        //HTR types 2 and 3 
        else if ((ieta == 5 && iside > 0) || ieta == 4)   fpga = "top";
        else if ((ieta == 5 || ieta >= 10) && iside < 0)  fpga = "bot";
        else if ((ieta < 10 && ieta >= 6) && iside < 0)   fpga = "top";
        else    edm::LogInfo( "HcalLogicalMapGenerator") <<"Bad fpga code"<<endl;
        
        //dphi
        if      (ieta <= 20) idphi = 1;
        else                 idphi = -1000;
        
        //create values usable in arrays from side and fpga
        if   (iside == 1) sidear = 1;
        else              sidear = 0;
        
        if (fpga == "bot") itb = 1;//convention different than for the
        else               itb = 0;//electronics id, modified in the 
	//MapEntry code

        phmod24 = iph % 24;
        
        //Again, x - 1 because the array starts at 0 while the variables start at 1
        irm = HO_RM_table[phmod24][iet][sidear];
        
        //x - 5 for the eta array for the same reason
        //  the new stuff for HX
        if          (ieta==16) irm_fi = HO_RM_fi_eta16;
        else if (ieta >= 5) irm_fi = HO_RM_fi_eta5to15[ieta - 5];
        else if (ieta <= 3) irm_fi = HO_RM_fi_eta1to4[iph][0][sidear];
        else if (ieta == 4) irm_fi = HO_RM_fi_eta1to4[iph][1][sidear];
        else                irm_fi = -1000;
        
        //Determine which of HTR in the set belongs here. It depends only on eta and side.
        //  the new stuff for HX
        //if          (ieta==16)
        //
        if (ieta <= 3 || (ieta >= 14 && iside == 1))     ih = 0;
        else if (ieta <= 13 && ieta >= 6 && iside == 1)  ih = 1;
        else if (ieta <= 13 && ieta >= 6 && iside == -1) ih = 3;
        else                                             ih = 2;
        
        //Each value of "is" covers 30 degrees (that is, 6 values of phi). To calculate which ones,
        //I use phi % 18. Crates start at phi = 71, 17, 35, 53
        
        if (iphi % 18 == 17 || iphi % 18 <= 4)      is = 0;
        else if (iphi % 18 >= 5 && iphi % 18 <= 10) is = 1;
        else                                        is = 2;
        
        if ( ieta == 16 && iside > 0 ) ihtr=21;
        else ihtr=ihslotho[is][ih];
        
        if ((ieta > 15 && iside > 0) && (icrate == 3 || icrate == 6))   ispigot = 12;
        else if ((ieta > 15 && iside > 0) && (icrate == 7 || icrate == 13))   ispigot = 13;
        else ispigot=ihtr<9?(ihtr-2)*2+itb:(ihtr-13)*2+itb;
        idcc=ihtr<9?1:2;
        idcc_sl = idcc == 1 ?10:20;
        
        ifed=fedhonum[ic][idcc-1];
        
        //HTR fiber
        
        //
        if (ieta >= 5 && phmod6e450) ihtr_fi = HO_htr_fi_450eta5to15[sidear][ieta - 5];
        else if (ieta >= 5 && phmod6e123) ihtr_fi = HO_htr_fi_450eta5to15[sidear][ieta - 5] - 1;
        else if (ieta == 4)               ihtr_fi = HO_htr_fi_eta4[sidear][phmod6];
        else if (ieta <= 3)               ihtr_fi = HO_htr_fi_eta123[sidear][phmod6];
        else                              ihtr_fi = -1000;
        
        //Fiber Channel
        //Eta >= 5 bools
        phi1458   = (iphi % 12 == 1 || iphi % 12 == 4 || iphi % 12 == 5  || iphi % 12 == 8);
        phi271011 = (iphi % 12 == 2 || iphi % 12 == 7 || iphi % 12 == 10 || iphi % 12 == 11);
        
        //Ring 0 bools
        phir0v1 = (iphi % 24 == 0 || iphi % 24 == 2 || iphi % 24 == 4  || iphi % 24 == 18 || iphi % 24 == 20 || iphi % 24 == 22);
        phir0v2 = (iphi % 24 == 1 || iphi % 24 == 3 || iphi % 24 == 17 || iphi % 24 == 19 || iphi % 24 == 21 || iphi % 24 == 23);
        //v3: phi 5 to 15 odd; v4: phi 6 to 16 even
        phir0v3 = (iphi % 24 == 5 || iphi % 24 == 7 || iphi % 24 == 9  || iphi % 24 == 11 || iphi % 24 == 13 || iphi % 24 == 15);
        phir0v4 = (iphi % 24 == 6 || iphi % 24 == 8 || iphi % 24 == 10 || iphi % 24 == 12 || iphi % 24 == 14 || iphi % 24 == 16);
        
        if (ieta >= 5){
          if      (ieta % 2 == 0 && phi1458)       ifi_ch = 0; 
          else if (ieta % 2 == 0 && iphi % 3 == 0) ifi_ch = 1;
          else if (ieta % 2 == 0 && phi271011)     ifi_ch = 2;
          else if (ieta % 2 == 1 && iphi % 3 == 0) ifi_ch = 0; 
          else if (ieta % 2 == 1 && phi271011)     ifi_ch = 1;
          else if (ieta % 2 == 1 && phi1458)       ifi_ch = 2;
        }
        else if (ieta == 4){
          if (iside == -1){
            if      (phir0v1)       ifi_ch = 0;
            else if (phir0v4)       ifi_ch = 1;
            else if (iphi % 2 == 1) ifi_ch = 2;
          }
          else{
            if      (phir0v3)       ifi_ch = 0;
            else if (phir0v2)       ifi_ch = 1;
            else if (iphi % 2 == 0) ifi_ch = 2;
          }
        }
        //eta = -3 and eta = +2
        else if ((ieta == 3 && iside == -1) || (ieta == 2 && iside == 1)){
          if      (phir0v4)            ifi_ch = 0;
          else if (phir0v3)            ifi_ch = 1;
          else if (phir0v1 || phir0v2) ifi_ch = 2;
        }
        //eta = -2 and eta = +3
        else if ((ieta == 3 && iside == 1) || (ieta == 2 && iside == -1)){
          if      (phir0v2)            ifi_ch = 0;
          else if (phir0v1)            ifi_ch = 1;
          else if (phir0v3 || phir0v4) ifi_ch = 2;
        }
        //ieta = 1
        else if (ieta == 1){
          if      (phir0v1 || phir0v3) ifi_ch = 0;
          else if (phir0v2 || phir0v4) ifi_ch = 1;
        }
        	
        //Intentional integer truncation; iqie and iadc are the same across all subdetectors
        //(Although irm_fi for HF starts at 1 and for HO it starts at 2, so one can't just copy and paste)
        iqie = (irm_fi - 2) / 2 + 1;
        
        if (irm_fi % 2 == 0) iadc = ifi_ch;
        else       	     iadc = NFCH + (ifi_ch + 1) % 3;
        
	//Pixel and Letter Code (Ring 0 is separate)
        if (ieta <= 4){
          ipixel = ipixelHO[irm_fi - 2][ifi_ch][0];
          letter = letterHO[irm_fi - 2][ifi_ch][0];
        }
        else{
          ipixel = ipixelHO[irm_fi - 2][ifi_ch][1];
          letter = letterHO[irm_fi - 2][ifi_ch][1];
        }
        
        //RBX and sector
        
        if (iside == -1) sidesign = 'M';
        else      	 sidesign = 'P';
        
        if      (ieta <= 4)                ring = 0;
        else if (ieta >= 5 && ieta <= 10)  ring = 1;
        else                               ring = 2;
        
        //Sector ranges from 1 to 12 depending on phi. Sector 1 goes 71,72,1,2,3,4 so I start at -2
        sector = 0;
        for (i = -2; i < iphi; i+=6){
          sector++;
        }
        if (sector > 12) sector = 1; //It rolls over for phi = 71,72 
        
        isector = sector;
        
        //For rings 1 and 2, we only want even sectors for the rbx
        if (ring != 0 && sector % 2 != 0) sector++;
        
        if (ring == 0)  sprintf (tempbuff, "%s%i%2.2d", det.c_str(), ring, sector);
        else            sprintf (tempbuff, "%s%i%c%2.2d", det.c_str(), ring, sidesign, sector);
        mystream<<tempbuff;
        rbx = mystream.str();
        mystream.str("");
        if (ieta == 16) {
	  det = "HOX";
	  letter = "X";}
        else det = "HO";
        
        //  the new htr_fi stuff for HX
        if (ieta==16 && isid<0) ihtr_fi = HO_htr_fi_450eta16M[irm - 1];
        else if (ieta==16 && isid>0) ihtr_fi = HO_htr_fi_450eta16P[irm - 1][sector/2-1];
        
        if ((ieta > 15 && iside > 0) && (icrate == 3 || icrate == 7))   {
          icrate = 6;
          ifed = 729;
        }
        else if ((ieta > 15 && iside > 0) && (icrate == 6 || icrate == 13))   {
          icrate = 7;
          ifed = 727;
        }
        
        if ( ieta == 16 ) ieta = 15;
        // fixing the switched RM's
        if ( iside == 1 && ring == 2 ) {
          if ( iphi == 5 ) {
            ieta = rmspecialeta[0][iet - 10];
            letter = rmspeciallet_code[0][iet - 10];
            det = rmspecialdet[0][iet - 10];
          }
          else if ( iphi == 67 ) {
            ieta = rmspecialeta[1][iet - 10];
            letter = rmspeciallet_code[1][iet - 10];
            det = rmspecialdet[1][iet - 10];
          }
        }
        else if ( iside == -1) {
          if ( ring == 1 ) {
            if ( iphi == 57 ) {
              ieta = rmspecialeta[2][iet - 4];
              letter = rmspeciallet_code[2][iet - 4];
              det = rmspecialdet[2][iet - 4];
            }
            else if ( iphi == 65 ) {
              ieta = rmspecialeta[3][iet - 4];
              letter = rmspeciallet_code[3][iet - 4];
              det = rmspecialdet[3][iet - 4];
            }
          }
          else if ( ring == 2 ) {
            if ( iphi == 16 ) {
              ieta = rmspecialeta[4][iet - 10];
              letter = rmspeciallet_code[4][iet - 10];
              det = rmspecialdet[4][iet - 10];
            }
            else if ( iphi == 31 ) {
              ieta = rmspecialeta[5][iet - 10];
              letter = rmspeciallet_code[5][iet - 10];
              det = rmspecialdet[5][iet - 10];
            }
          }
        }

        HOHXLogicalMapEntry hoxlmapentry(
					 ifi_ch, ihtr_fi, ispigot, ifed, icrate, ihtr, fpga,
					 det, iside, ieta, iphi, idepth,
					 idphi, isector, irm, irm_fi, ipixel, iqie, iadc,
					 rbx, letter
					 );
	HOHXEntries.push_back(hoxlmapentry);
	LinearIndex2Entry.at(hoxlmapentry.getLinearIndex())=HcalLogicalMap::makeEntryNumber(1,1,HOHXEntries.size()-1);

	const HcalGenericDetId hgdi(hoxlmapentry.getDetId());
	unsigned int denseId;
	if (hgdi.genericSubdet()==HcalGenericDetId::HcalGenOuter) {
	  denseId=topo->detId2denseIdHO(hgdi);
	  HoHash2Entry.at(denseId)=HOHXEntries.size();
	} else if (hgdi.genericSubdet()==HcalGenericDetId::HcalGenCalibration) {
	  denseId=topo->detId2denseIdCALIB(hgdi);
	  HxCalibHash2Entry.at(denseId)=HOHXEntries.size();
	}

        if (ring==0){
          if (ipixel==1){
            ipixel = 2;
            iadc = 4;
            ifi_ch = 0;
            letter = "X";
            det = "HOX";
            HOHXLogicalMapEntry hoxlmapentry(
					     ifi_ch, ihtr_fi, ispigot, ifed, icrate, ihtr, fpga,
					     det, iside, ieta, iphi, idepth,
					     idphi, isector, irm, irm_fi, ipixel, iqie, iadc,
					     rbx, letter
					     );
	    HOHXEntries.push_back(hoxlmapentry);
	    LinearIndex2Entry.at(hoxlmapentry.getLinearIndex())=HcalLogicalMap::makeEntryNumber(1,1,HOHXEntries.size()-1);

	    const HcalGenericDetId hgdi(hoxlmapentry.getDetId());
	    unsigned int denseId;
	    if (hgdi.genericSubdet()==HcalGenericDetId::HcalGenOuter) {
	      denseId=topo->detId2denseIdHO(hgdi);
	      HoHash2Entry.at(denseId)=HOHXEntries.size();
	    } else if (hgdi.genericSubdet()==HcalGenericDetId::HcalGenCalibration) {
	      denseId=topo->detId2denseIdCALIB(hgdi);
	      HxCalibHash2Entry.at(denseId)=HOHXEntries.size();
	    }
          }
          else if (ipixel==17){
            ipixel = 18;
            iadc = 1;
            ifi_ch = 1;
            letter = "X";
            det = "HOX";
            HOHXLogicalMapEntry hoxlmapentry(
					     ifi_ch, ihtr_fi, ispigot, ifed, icrate, ihtr, fpga,
					     det, iside, ieta, iphi, idepth,
					     idphi, isector, irm, irm_fi, ipixel, iqie, iadc,
					     rbx, letter
					     );
	    HOHXEntries.push_back(hoxlmapentry);
	    LinearIndex2Entry.at(hoxlmapentry.getLinearIndex())=HcalLogicalMap::makeEntryNumber(1,1,HOHXEntries.size()-1);

	    const HcalGenericDetId hgdi(hoxlmapentry.getDetId());
	    unsigned int denseId;
	    if (hgdi.genericSubdet()==HcalGenericDetId::HcalGenOuter) {
	      denseId=topo->detId2denseIdHO(hgdi);
	      HoHash2Entry.at(denseId)=HOHXEntries.size();
	    } else if (hgdi.genericSubdet()==HcalGenericDetId::HcalGenCalibration) {
	      denseId=topo->detId2denseIdCALIB(hgdi);
	      HxCalibHash2Entry.at(denseId)=HOHXEntries.size();
	    }
          }
        }
      }
    }
  }
} 

void HcalLogicalMapGenerator::buildCALIBMap(const HcalTopology* topo,
					    std::vector <CALIBLogicalMapEntry>& CALIBEntries,
					    std::vector <uint32_t>& LinearIndex2Entry,
					    std::vector <uint32_t>& HxCalibHash2Entry) {
  
  /******************************/
  /* CALIB crate numbering in order of FEDID*/
  int calibcrate_loc[NCALIBCR]={4,0,1,5,11,15,17,14,10,9,7,6,13};//HBHE,HF,HO
  memcpy( calibcrate, calibcrate_loc, sizeof(int)*NCALIBCR );
  /* CALIB FED numbering of DCCs */
  int fedcalibnum_loc[NCALIBCR][2]={{700,701},{702,703},{704,705},{706,707},{708,709},{710,711},{712,713},{714,715},{716,717},/*calib_hbhe 4,0,1,5,11,15,17,14,10*/
				    {720,721},/*calib_hf 9*/
				    {726,727},{728,729},{730,731}};//calib_ho 7,6,13
  memcpy( fedcalibnum, fedcalibnum_loc, sizeof(int)*NCALIBCR*2 );
  /* iphi (lower) starting index for each CALIB crate */
  int icalibphis_loc[NCALIBCR]={3,11,19,27,35,43,51,59,67,27,17,35,53};
  memcpy( icalibphis, icalibphis_loc, sizeof(int)*NCALIBCR );
  /* icalibsector_min minimum sector on each crate {0,1,4,5,10,11,14,15,17,9,7,6,13}*/
  int icalibsector_min_loc[NCALIBCR]={2,4,6,8,10,12,14,16,18,1,4,7,10};
  memcpy( icalibsector_min, icalibsector_min_loc, sizeof(int)*NCALIBCR );
  /* icalibsector_max maximum sector on each crate */
  int icalibsector_max_loc[NCALIBCR]={3,5,7,9,11,13,15,17,1,4,6,9,12};
  memcpy( icalibsector_max, icalibsector_max_loc, sizeof(int)*NCALIBCR );
  //detectors represented on each crate (0 for top, 1 for bot)
  std::string detIDCALIB_loc[NCALIBCR][NTOPBOT] = {{"HB","HE"},/*crate 4*/ 
						   {"HB","HE"},/*crate 0*/
						   {"HB","HE"},/*crate 1*/
						   {"HB","HE"},/*crate 5*/
						   {"HB","HE"},/*crate 11*/
						   {"HB","HE"},/*crate 15*/
						   {"HB","HE"},/*crate 17*/
						   {"HB","HE"},/*crate 14*/
						   {"HB","HE"},/*crate 10*/
						   {"HF","HF"},/*crate 9*/
						   {"HO","HO"},/*crate 7*/
						   {"HO","HO"},/*crate 6*/
						   {"HO","HO"}};/*crate 13*/
  for (int jj = 0; jj < NCALIBCR; jj++) {
    for (int kk = 0; kk < NTOPBOT; kk++) {
      detIDCALIB[jj][kk] = detIDCALIB_loc[jj][kk];
    } 
  }
                                                         
  int nfbr_max_loc[NCALIBCR][NTOPBOT] = {{4,8},{4,8},{4,8},{4,8},{4,8},{4,8},{4,8},{4,8},{4,8},{4,4},{8,4},{8,8},{8,4}};
  memcpy( nfbr_max, nfbr_max_loc, sizeof(int)*NCALIBCR*NTOPBOT );
  //HB only utilizes 4 htr_fibers HE all 8
  int calibHOinfo_loc[NHOCR][NTOPBOT][NFBR][2]={/*sector and ring of HO for the calib channel*/
    {{{4,0},{5,0},{6,0},{},{6,-2},{6,-1},{6,1},{6,2}},
     {{4,-2},{4,-1},{4,1},{4,2},{},{},{},{}}},/*crate 7*/

    {{{7,0},{8,0},{9,0},{},{8,-2},{8,-1},{8,1},{8,2}},
     {{1,0},{2,0},{3,0},{},{2,-2},{2,-1},{2,1},{2,2}}},/*crate 6*/

    {{{10,0},{11,0},{12,0},{},{10,-2},{10,-1},{10,1},{10,2}},
     {{12,-2},{12,-1},{12,1},{12,2},{},{},{},{}}}/*crate 13*/
  };
  memcpy( calibHOinfo, calibHOinfo_loc, sizeof(int)*NHOCR*NTOPBOT*NFBR*2 );

  /*********************************/

  //Stream variable

  stringstream mystream;
  /**/
  ihtr=8; //calibration units are housed on htr slot 8
  idcc=1;
  idcc_sl=10;
  irm_fi = 1;// everything other than HE is on A
  det = "";

  /*****************************************************/
  /*HBHE calibration channels*/
  for(ic=0; ic<NHBHECR; ic++){
    icrate=calibcrate[ic];
    ifed=fedcalibnum[ic][idcc-1];
    /* top and bottom */
    for(itb=0; itb<NTOPBOT; itb++){
      /* 4 or 8 fibers used per HTR FPGA */
      for(ifb=0; ifb<nfbr_max[ic][itb]; ifb++){
        det = detIDCALIB[ic][itb];
        ihtr_fi=ifb+1;
        irm_fi = 1;// everything other than he is on A
        idphi=4;
        if (itb==1) if(ihtr_fi%2==0) irm_fi = 2;  //all of the even fibers in HE are on B rather than A
        if (det=="HE"){
          ispigot=13;
          fpga="bot";
          if (ihtr_fi==1||ihtr_fi==2||ihtr_fi==5||ihtr_fi==6) {
            iwedge = icalibsector_min[ic];
          }
          else if (ihtr_fi==3||ihtr_fi==4||ihtr_fi==7||ihtr_fi==8) {
            iwedge = icalibsector_max[ic];
          }
          if (ihtr_fi<5) {
            sidesign = 'M';
            S_side = '1';
            ieta = -1;
            iside=-1;
          }
          else {
            sidesign = 'P';
            S_side = '1';
            ieta = 1;
            iside=1;
          }
        }
        else if (det=="HB"){
          ispigot=12;
          fpga="top";
          (ihtr_fi%2==1) ? iwedge = icalibsector_min[ic] : iwedge = icalibsector_max[ic];
          if (ihtr_fi<3) {
            sidesign = 'M';
            S_side = '1';
            ieta = -1;
            iside=-1;
          }
          else {
            sidesign = 'P';
            S_side = '1';
            ieta = 1;
            iside=1;
          }
        }
        iphi = ((iwedge*idphi) + 71 - idphi)%72;
        subdet = "CALIB_"+det;
        sprintf (tempbuff, "%s%c%2.2i%c", det.c_str(), sidesign, iwedge,'\0');
        mystream<<tempbuff;
        rbx = mystream.str();
        mystream.str("");
        /* three channels per fiber */
        for(ifc=0; ifc<NFCH; ifc++){
          ifi_ch=ifc;
          if (irm_fi==1){
            if (ifc==0) ich_type=0;
            else if (ifc==1) ich_type=1;
            else if (ifc==2) {
              if (det=="HB") ich_type=2;
              else if(det=="HE") ich_type=3;
            }
          }
          else if (irm_fi==2){
            if (ifc==0) ich_type=4;
            else if (ifc==1) ich_type=5;
            else if (ifc==2) ich_type=6;
          }
          CALIBLogicalMapEntry caliblmapentry(
					      ifi_ch, ihtr_fi, ispigot, ifed, icrate, ihtr, fpga,
					      det, ieta, iphi, ich_type, 
					      iside, idphi, rbx, iwedge, irm_fi,
					      subdet
					      );
	  CALIBEntries.push_back(caliblmapentry);
	  LinearIndex2Entry.at(caliblmapentry.getLinearIndex())=HcalLogicalMap::makeEntryNumber(1,2,CALIBEntries.size()-1);

	  const HcalGenericDetId hgdi(caliblmapentry.getDetId());	  
	  const unsigned int hashedId=topo->detId2denseIdCALIB(hgdi);
	  if (hgdi.genericSubdet()==HcalGenericDetId::HcalGenCalibration) HxCalibHash2Entry.at(hashedId)=CALIBEntries.size();
        }
      }
    }
  }
  
  /* only one CALIB HF crate */
  /* top and bottom */
  for(itb=0; itb<NTOPBOT; itb++){
    /* four CALIB HF fibers per HTR FPGA */
    for(ifb=0; ifb<4; ifb++){
      /* three channels per fiber */
      for(ifc=0; ifc<NFCH; ifc++){
        icrate=calibcrate[ic];
        det = "HF";
        ihtr_fi=ifb+1;
        ifi_ch=ifc;
        (ihtr_fi==1)?iphi=1:((ihtr_fi==2)?iphi=19:((ihtr_fi==3)?iphi=37:iphi=55));
        idphi=18;
        ifed=fedcalibnum[ic][idcc-1];
        (ifc==0)?ich_type=8:(ifc==1?ich_type=0:ich_type=1);
	//changed ch_type of fibre channel 0 from 2 to 8, as per HcalCalibDetId specification
        irm_fi = 1;
	//Wedge has steps of 3, HF(P/M)2,5,8,11
        //iwedge=ihtr_fi;
	iwedge=2+(ifb*3);
        if (itb==0){
          ispigot=12;
          fpga="top";
          sidesign='P';
          iside=1;
          ieta=1;
        }
        else {
          ispigot=13;
          fpga="bot";
          sidesign='M';
          iside=-1;
          ieta=-1;
        }
        subdet = "CALIB_"+det;
        sprintf (tempbuff, "%s%c%2.2i%c", det.c_str(), sidesign, iwedge,'\0');
        mystream<<tempbuff;
        rbx = mystream.str();
        mystream.str("");
        CALIBLogicalMapEntry caliblmapentry(
					    ifi_ch, ihtr_fi, ispigot, ifed, icrate, ihtr, fpga,
					    det, ieta, iphi, ich_type, 
					    iside, idphi, rbx, iwedge, irm_fi,
					    subdet
					    );
	CALIBEntries.push_back(caliblmapentry);
	LinearIndex2Entry.at(caliblmapentry.getLinearIndex())=HcalLogicalMap::makeEntryNumber(1,2,CALIBEntries.size()-1);

	const HcalGenericDetId hgdi(caliblmapentry.getDetId());	  
	const unsigned int hashedId=topo->detId2denseIdCALIB(hgdi);
	if (hgdi.genericSubdet()==HcalGenericDetId::HcalGenCalibration) HxCalibHash2Entry.at(hashedId)=CALIBEntries.size();

      }
    }
  }
  ic++;

  /*HO calibration channels*/
  for(ic=ic; ic<NCALIBCR; ic++){
    icrate=calibcrate[ic];
    irm_fi = 1;// everything other than he is on A
    ifed=fedcalibnum[ic][idcc-1];
    /* top and bottom */
    for(itb=0; itb<NTOPBOT; itb++){
      det=detIDCALIB[ic][itb];
      /* 4 or 8 fibers used per HTR FPGA */
      for(ifb=0; ifb<nfbr_max[ic][itb]; ifb++){
        if (itb==1) {
          ispigot = 13;
          fpga = "bot";
        }
        else{
          ispigot = 12;
          fpga = "top";
        }
        ihtr_fi=ifb+1;
        iwedge=calibHOinfo[ic-10][itb][ifb][0];
        ieta=calibHOinfo[ic-10][itb][ifb][1];
        if (ieta<0) {
          iside=-1;
          sidesign='M';
        }
        else if (ieta>0){
          iside=1;
          sidesign='P';
        }
        else {
	  iside=0;
	  sidesign='0';
        }
        if (ieta==-2) S_side='2';
        else if (ieta==-1) S_side='1';
        else if (ieta==1) S_side='1';
        else if (ieta==2) S_side='2';

        subdet ="CALIB_"+det;
        if (ieta==0) sprintf (tempbuff, "%s%c%2.2i%c", det.c_str(), sidesign, iwedge,'\0');
        else  sprintf (tempbuff, "%s%c%c%2.2i%c", det.c_str(), S_side, sidesign, iwedge,'\0');
        mystream<<tempbuff;
        rbx = mystream.str();
        mystream.str("");
        /* only two channels used in HO per fiber */
        //now new information suggests that the third channel is used for x-talk
        //but only in some of the rm's, seems to be dependent on whether nfbr_max
        //sector 10 on YB+2,1,0,-1, sector 12 on YB-2
        int NFCH_HO;
        //( nfbr_max[ic][itb] == 4 ) ? NFCH_HO = 2 : NFCH_HO = 3;
        ( ieta == -2 ) ? ( iwedge == 12 ? NFCH_HO = 3 : NFCH_HO = 2 ) : ( iwedge == 10) ? NFCH_HO = 3 : NFCH_HO = 2;
        for(ifc=0; ifc<NFCH_HO; ifc++){
          ifi_ch=ifc;
          (ifi_ch == 2) ? ich_type = 7 : ich_type = ifi_ch;
          (ieta==0) ? idphi = 6 : idphi = 12;
          
          (ieta==0) ? iphi=((iwedge*idphi)+71-idphi)%72 : iphi=(((iwedge/2)*idphi)+71-idphi)%72;
          //nothing on htr_fi=4 for the top
          do {
            if (iside==0&&ifb==3) break;
            CALIBLogicalMapEntry caliblmapentry(
						ifi_ch, ihtr_fi, ispigot, ifed, icrate, ihtr, fpga,
						det, ieta, iphi, ich_type, 
						iside, idphi, rbx, iwedge, irm_fi,
						subdet
						);
	    CALIBEntries.push_back(caliblmapentry);
	    LinearIndex2Entry.at(caliblmapentry.getLinearIndex())=HcalLogicalMap::makeEntryNumber(1,2,CALIBEntries.size()-1);

	    const HcalGenericDetId hgdi(caliblmapentry.getDetId());	  
	    const unsigned int hashedId=topo->detId2denseIdCALIB(hgdi);
	    if (hgdi.genericSubdet()==HcalGenericDetId::HcalGenCalibration) HxCalibHash2Entry.at(hashedId)=CALIBEntries.size();
	  } while (ifb!=ifb);
        }
      }
    }
  }
} 

void HcalLogicalMapGenerator::buildZDCMap(const HcalTopology* topo, std::vector <ZDCLogicalMapEntry>& ZDCEntries,
					  std::vector <uint32_t>& LinearIndex2Entry,
					  std::vector <uint32_t>& ZdcHash2Entry) {
  /******************************/
  /*ZDC channels*/
  //Stream variable
  stringstream mystream;

  ifed=722;
  iy=1;
  ihtr=8;
  icrate=12;
  idcc=1;
  idcc_sl=10;
  ispigot=12;
  /* side plus and minus */
  for(itb=0; itb<NTOPBOT; itb++){
    if (itb==0) {
      iside = 1;
      irm = 1;
      fpga = "top";
    }
    else {
      iside = -1;
      irm = 2;
      if (mapIOV_<4) {
	fpga = "top";
	ispigot = 12;}
      else {
	fpga = "bot";
	ispigot = 13;}
    }
    /*loop over ZDC cables*/
    for(icab=1; icab<NZDCCAB+1; icab++){
      if (icab < 4) {
        irm_fi = 1;
        iadc = icab - 1;
        ifi_ch = iadc;	
      }
      else if (icab < 7) {
        irm_fi = 2;
        iadc = icab - 1;
        ifi_ch = iadc - 3;
      }
      else {
        irm_fi = 3;
        iadc = icab - 7;
        ifi_ch = iadc;
      }
      if (itb==0) {
        ihtr_fi = irm_fi;
      }
      else {
	if (mapIOV_<4) ihtr_fi = irm_fi + 3;
        else ihtr_fi = irm_fi;
      }
      if (icab < 6) {
        ix = icab;
        idepth = 1;
        idx = 1;
        det = "ZDC_EM";
        idet_ch = ix;
        iqie = 1;
      }
      else {
        ix = 1;
        idepth = icab - 4;
        idx = 5;
        det = "ZDC_HAD";
        idet_ch = icab - 5;
        if (icab==6) iqie = 1;
        else iqie =  2;
      }
      ZDCLogicalMapEntry zdclmapentry(
				      ifi_ch, ihtr_fi, ispigot, ifed, icrate, ihtr, fpga,
				      det, iside, idepth, 
				      ix, iy, idx, idet_ch, icab, irm, iqie, 
				      iadc, irm_fi
				      );
      ZDCEntries.push_back(zdclmapentry);
      LinearIndex2Entry.at(zdclmapentry.getLinearIndex())=HcalLogicalMap::makeEntryNumber(1,3,ZDCEntries.size()-1);
    }
  }
}


/********************************************************/
void HcalLogicalMapGenerator::ConstructTriggerTower(const HcalTopology* topo,
						    std::vector <HTLogicalMapEntry>& HTEntries,
						    int iside, int ieta, int iphi, int idphi, int idepth, std::string det, int iwedge, int irm,
						    int ipixel, int iqie, int iadc, int irm_fi, int ifi_ch, int icrate, int ihtr, std::string fpga,
						    int ihtr_fi, int ispigot, int islb, std::string slbin, std::string slbin2,
						    std::string slnam, int irctcra, int irctcar, int irctcon, std::string rctnam, int ifed
						    )
{
	
  string t_chDet,t_fpga;
  string t_slnam,t_rctnam,t_slbin,t_slbin2;

  int t_nDat=-1;
  int t_side,t_iEta,t_iPhi,t_jPhi,t_iDep,t_topbot;
  int t_wedge,t_crate,t_htr;
  //  int t_rm,t_pixel,t_qie,t_adc,t_rm_fi,t_fi_ch,t_htr_fi;
  int t_spigo,t_slb,t_rctcra,t_rctcar,t_rctcon,t_fedid;

  //making global variables local for modification
  (fpga=="bot") ? t_topbot = 1 : t_topbot = 0;
  t_side   = iside;
  t_iEta   = ieta;
  t_iPhi   = iphi;
  t_jPhi   = idphi;
  t_iDep   = idepth;
  t_chDet  = det;
  t_wedge  = iwedge;
  //  t_rm     = irm;
  //  t_pixel  = ipixel;
  //  t_qie    = iqie;
  //  t_adc    = iadc;
  //  t_rm_fi  = irm_fi;
  //  t_fi_ch  = ifi_ch;
  t_crate  = icrate;
  t_htr    = ihtr;
  t_fpga   = fpga;
  //  t_htr_fi = ihtr_fi;
  t_spigo  = ispigot;
  t_slb    = islb;
  t_slbin  = slbin;
  t_slbin2 = slbin2;
  t_slnam  = slnam;
  t_rctcra = irctcra;
  t_rctcar = irctcar;
  t_rctcon = irctcon;
  t_rctnam = rctnam;
  t_fedid  = ifed;

  do {
    if(t_iDep!=1) break;
    if(t_chDet=="HE"&&t_iEta==29) break;

    if(t_chDet=="HF") {
      t_jPhi=4;//20 degree slices in HF
      if(t_iEta==30||t_iEta==31||t_iEta==33||
	 t_iEta==34||t_iEta==36||t_iEta==37||
         t_iEta==39||t_iEta==40||t_iEta==41) break;

      else if(t_iEta==32)t_iEta=30;
      else if(t_iEta==35)t_iEta=31;
      else if(t_iEta==38)t_iEta=32;

      if(t_iPhi==3||t_iPhi==7||t_iPhi==11||t_iPhi==15||t_iPhi==19||
	 t_iPhi==23||t_iPhi==27||t_iPhi==31||t_iPhi==35||t_iPhi==39||
	 t_iPhi==43||t_iPhi==47||t_iPhi==51||t_iPhi==55||t_iPhi==59||
	 t_iPhi==63||t_iPhi==67||t_iPhi==71) break;
    }
  
    if(t_side<0) t_iEta=-t_iEta;
    t_chDet="HT";
    
    if(t_slbin!="NA") {
      if(t_slbin2!="NA") t_jPhi=1;
      if(t_slbin=="A0"||t_slbin=="B0")  t_nDat=0;
      else if(t_slbin=="A1"||t_slbin=="B1")  t_nDat=1;
      else if(t_slbin=="C0"||t_slbin=="D0")  t_nDat=2;
      else if(t_slbin=="C1"||t_slbin=="D1")  t_nDat=3;
      
      HTLogicalMapEntry htlmapentry(
				    t_iEta, t_iPhi,
				    t_side, t_jPhi, t_iDep, t_chDet, t_wedge, t_crate, t_htr, t_topbot, 
				    t_spigo, t_slb, t_slbin, t_slbin2, t_nDat,
				    t_slnam, t_rctcra, t_rctcar, t_rctcon, t_rctnam, t_fedid
				    );
      HTEntries.push_back(htlmapentry);
    }
    if(t_slbin2!="NA") {
      if(t_slbin2=="A0"||t_slbin2=="B0")  t_nDat=0;
      else if(t_slbin2=="A1"||t_slbin2=="B1")  t_nDat=1;
      else if(t_slbin2=="C0"||t_slbin2=="D0")  t_nDat=2;
      else if(t_slbin2=="C1"||t_slbin2=="D1")  t_nDat=3;
      t_jPhi=1;//after splitting up these towers, each half should have dphi of 1
      t_iPhi+=1;

      HTLogicalMapEntry htlmapentry(
				    t_iEta, t_iPhi,//changed from t_iPhi + 1 here to the spot above
				    t_side, t_jPhi, t_iDep, t_chDet, t_wedge, t_crate, t_htr, t_topbot, 
				    t_spigo, t_slb, t_slbin, t_slbin2, t_nDat,
				    t_slnam, t_rctcra, t_rctcar, t_rctcon, t_rctnam, t_fedid
				    );
      HTEntries.push_back(htlmapentry);
    }
  } while(1!=1);
}
