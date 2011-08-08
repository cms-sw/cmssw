#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalOtherDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "CondFormats/HcalObjects/interface/HcalMappingEntry.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string>
#include <cstring>

using namespace std;


//class HBHEHFLogicalMapEntry 

HBHEHFLogicalMapEntry::HBHEHFLogicalMapEntry( 
             int in_fi_ch, int in_htr_fi, int in_spig, int in_fed, int in_cr, int in_htr, std::string in_s_fpga,
             std::string in_s_det, int in_sid, int in_et, int in_ph, int in_dep,
             int in_dph, int in_wed, int in_rm, int in_rm_fi, int in_pix, int in_qie, int in_adc,
             int in_slb, int in_rctcra, int in_rctcar, int in_rctcon,
             std::string in_s_rbx, std::string in_s_slb, std::string in_s_slb2, std::string in_s_slnam, std::string in_s_rctnam
  )
{
  int mytopbot;
  (in_s_fpga=="bot") ? mytopbot = 0 : mytopbot = 1;

  mydphi_   = in_dph;
  mywedge_  = in_wed;
  myslb_    = in_slb;
  myrctcra_ = in_rctcra;
  myrctcar_ = in_rctcar;
  myrctcon_ = in_rctcon;

  // string data members
  myslbin_  = in_s_slb;
  myslbin2_ = in_s_slb2;
  myslnam_  = in_s_slnam;
  myrctnam_ = in_s_rctnam;

  // create the hcal electronics id
  HcalElectronicsId heid( in_fi_ch, in_htr_fi, in_spig, in_fed - 700 );
  heid.setHTR( in_cr, in_htr, mytopbot );
  // create the hcal detector id
  HcalSubdetector mysubdet          = HcalEmpty;
  if (in_s_det=="HB") mysubdet      = HcalBarrel;
  else if (in_s_det=="HE") mysubdet = HcalEndcap;
  else if (in_s_det=="HF") mysubdet = HcalForward;

  HcalDetId hdid( mysubdet, in_sid*in_et, in_ph, in_dep );

  HcalFrontEndId hrbx( in_s_rbx, in_rm, in_pix, in_rm_fi, in_fi_ch, in_qie, in_adc );

  // store the different ids
  hcalEID_        = heid.rawId();
  hcalDetID_      = hdid.rawId();
  hcalFrontEndID_ = hrbx.rawId();
}

char* HBHEHFLogicalMapEntry::printLMapLine() {

  static char myline[512];

  HcalElectronicsId hcaleid(hcalEID_);
  HcalDetId         hcaldid(hcalDetID_);
  HcalGenericDetId  hcalgenid(hcalDetID_);
  HcalFrontEndId    rbxid(hcalFrontEndID_);

  int mydcc_sl = 0;
  int mydcc    = 0;
  if ((hcaleid.dccid()%2)==1) {
    mydcc_sl = 19;
    mydcc    = 2;
  } 
  else {
    mydcc_sl = 9;
    mydcc    = 1;
  } 

  string myfpga = "";
  string mydet  = "";
  HcalSubdetector mysubdet = hcaldid.subdet();
  (mysubdet==1) ? mydet = "HB" : 
              ((mysubdet==2) ? mydet = "HE" :
              ((mysubdet==3) ? mydet = "HO" : 
              ((mysubdet==4) ? mydet = "HF" : 
              mydet = "invalid")));

  ( hcaleid.htrTopBottom()==0 ) ? myfpga = "bot" : myfpga = "top";

  sprintf(myline,"%1d %6d %6d %6d %6d %6d %6s %7s %6d %6d %6d",0,hcaldid.zside(),hcaldid.ietaAbs(),hcaldid.iphi(),mydphi_,hcaldid.depth(),mydet.c_str(),rbxid.rbx().c_str(),mywedge_,rbxid.rm(),rbxid.pixel());
  sprintf(myline+strlen(myline),"%6d %6d %6d %6d %6d %6d %6s",rbxid.qieCard(),rbxid.adc(),rbxid.rmFiber(),hcaleid.fiberChanId(),hcaleid.readoutVMECrateId(),hcaleid.htrSlot(),myfpga.c_str());
  sprintf(myline+strlen(myline),"%8d %7d %6d %6d %6d %6s",hcaleid.fiberIndex(),mydcc_sl,hcaleid.spigot(),mydcc,myslb_,myslbin_.c_str());
  sprintf(myline+strlen(myline),"%8s %15s    %6d %6d %6d %20s    %6d\n",myslbin2_.c_str(),myslnam_.c_str(),myrctcra_,myrctcar_,myrctcon_,myrctnam_.c_str(),hcaleid.dccid()+700);

  return myline;
}


// class HOHXLogicalMapEntry 

HOHXLogicalMapEntry::HOHXLogicalMapEntry(
           int in_fi_ch, int in_htr_fi, int in_spig, int in_fed, int in_cr, int in_htr, std::string in_s_fpga,
           std::string in_s_det, int in_sid, int in_et, int in_ph, int in_dep,
           int in_dph, int in_sec, int in_rm, int in_rm_fi, int in_pix, int in_qie, int in_adc,
           std::string in_s_rbx, std::string in_s_let 
  )
{
  int mytopbot;
  (in_s_fpga=="bot") ? mytopbot = 0 : mytopbot = 1;

  mydphi_   = in_dph;
  mysector_ = in_sec;
  // string data members
  myletter_ = in_s_let;

  // create the hcal electronics id
  HcalElectronicsId heid( in_fi_ch, in_htr_fi, in_spig, in_fed - 700 );
  heid.setHTR( in_cr, in_htr, mytopbot );

  if (in_s_det=="HO") {
    //create the hcal det id in the case of regular HO channel
    HcalDetId hdid( HcalOuter, in_sid*in_et, in_ph, in_dep );
    hcalDetID_ = hdid.rawId();
  }
  else {
    //create the calib det id in the case of HO cross talk channels
    HcalCalibDetId hdid( in_sid*in_et, in_ph );
    hcalDetID_ = hdid.rawId();
  }

  HcalFrontEndId hrbx( in_s_rbx, in_rm, in_pix, in_rm_fi, in_fi_ch, in_qie, in_adc );

  // store the different ids
  hcalEID_        = heid.rawId();
  hcalFrontEndID_ = hrbx.rawId();

}
  
char* HOHXLogicalMapEntry::printLMapLine() {

  static char myline[512];

  HcalElectronicsId hcaleid(hcalEID_);
  HcalGenericDetId hcalgenid(hcalDetID_);
  HcalFrontEndId rbxid(hcalFrontEndID_);

  int mydcc_sl = 0;
  int mydcc    = 0;
  if ((hcaleid.dccid()%2)==1) {
    mydcc_sl = 19;
    mydcc    = 2;
  } 
  else {
    mydcc_sl = 9;
    mydcc    = 1;
  } 

  string myfpga = "";
  string mydet  = "";
  int mydepth   = 0;
  int myside    = -2;
  int myeta     = 0;
  int myphi     = -1;
  if ( hcalgenid.isHcalCalibDetId() ) {
    HcalCalibDetId hcalcompid(hcalDetID_);
    mydet   = "HOX";
    mydepth = 4;
    myside  = hcalcompid.zside();
    myeta   = hcalcompid.ieta()*myside;
    myphi   = hcalcompid.iphi();
  }
  else if ( hcalgenid.isHcalDetId() ) {
    HcalDetId hcalcompid(hcalDetID_);
    HcalSubdetector mysubdet = hcalcompid.subdet();
    (mysubdet==HcalBarrel) ? mydet = "HB" : 
                ((mysubdet==HcalEndcap) ? mydet = "HE" :
                ((mysubdet==HcalOuter) ? mydet = "HO" : 
                ((mysubdet==HcalForward) ? mydet = "HF" : 
                mydet = "invalid")));
    mydepth = hcalcompid.depth();
    myside  = hcalcompid.zside();  
    myeta   = hcalcompid.ietaAbs();
    myphi   = hcalcompid.iphi();
  }

  ( hcaleid.htrTopBottom()==0 ) ? myfpga = "bot" : myfpga = "top";

  sprintf(myline,"%1d %6d %6d %6d %6d %6d %6s %7s %6d %6d %6d",0,myside,myeta,myphi,mydphi_,mydepth,mydet.c_str(),rbxid.rbx().c_str(),mysector_,rbxid.rm(),rbxid.pixel());
  sprintf(myline+strlen(myline),"%6d %6d %6d %6d %8s %6d %6d %6s",rbxid.qieCard(),rbxid.adc(),rbxid.rmFiber(),hcaleid.fiberChanId(),myletter_.c_str(),hcaleid.readoutVMECrateId(),hcaleid.htrSlot(),myfpga.c_str());
  sprintf(myline+strlen(myline),"%8d %7d %6d %6d %6d\n",hcaleid.fiberIndex(),mydcc_sl,hcaleid.spigot(),mydcc,hcaleid.dccid()+700);

  return myline;
}


// class CalibLogicalMapEntry 

CALIBLogicalMapEntry::CALIBLogicalMapEntry(
            int in_fi_ch, int in_htr_fi, int in_spig, int in_fed,  int in_cr, int in_htr, std::string in_s_fpga,  
            std::string in_s_det, int in_et, int in_ph, int in_ch_ty, 
            int in_sid, int in_dph, std::string in_s_rbx, int in_wed, int in_rm_fi,
            std::string in_s_subdet
  )
{
  int mytopbot;
  (in_s_fpga=="bot") ? mytopbot = 0 : mytopbot = 1;

  myside_   = in_sid;
  mydphi_   = in_dph;
  mywedge_  = in_wed;
  // string data members
  mycalibsubdet_ = in_s_subdet;

  //create the hcal electronics id
  HcalElectronicsId heid( in_fi_ch, in_htr_fi, in_spig, in_fed - 700 );
  heid.setHTR( in_cr, in_htr, mytopbot );

  //create the hcal det id for a calibration unit channel
  HcalSubdetector mysubdet = HcalEmpty;
  if (in_s_det=="HB") mysubdet = HcalBarrel;
  else if (in_s_det=="HE") mysubdet = HcalEndcap;
  else if (in_s_det=="HO") mysubdet = HcalOuter;
  else if (in_s_det=="HF") mysubdet = HcalForward;

  HcalCalibDetId hcalibdid( mysubdet, in_et, in_ph, in_ch_ty );

  int in_rm, in_pix, in_qie, in_adc;
  in_rm  = 5;
  in_pix = 0;
  in_qie = 1;
  in_adc = in_fi_ch + ( 3 * ( in_rm_fi - 1 ) );

  HcalFrontEndId hrbx( in_s_rbx, in_rm, in_pix, in_rm_fi, in_fi_ch, in_qie, in_adc );

  //store the different ids
  hcalEID_        = heid.rawId();
  hcalCalibDetID_ = hcalibdid.rawId();
  hcalFrontEndID_ = hrbx.rawId();
}

char* CALIBLogicalMapEntry::printLMapLine() {

  static char myline[512];

  HcalElectronicsId hcaleid(hcalEID_);
  HcalCalibDetId    hcalcalibid(hcalCalibDetID_);
  HcalGenericDetId  hcalgenid(hcalCalibDetID_);
  HcalFrontEndId    rbxid(hcalFrontEndID_);

  int mydcc_sl = 0;
  int mydcc    = 0;
  if ((hcaleid.dccid()%2)==1) {
    mydcc_sl = 19;
    mydcc    = 2;
  } 
  else {
    mydcc_sl = 9;
    mydcc    = 1;
  } 

  string myfpga = "";
  string mydet  = "";
  HcalSubdetector mysubdet = hcalcalibid.hcalSubdet();
  (mysubdet==HcalBarrel) ? mydet = "HB" : 
              ((mysubdet==HcalEndcap)  ? mydet = "HE" :
              ((mysubdet==HcalOuter)   ? mydet = "HO" : 
              ((mysubdet==HcalForward) ? mydet = "HF" : 
              mydet = "invalid")));
  (hcaleid.htrTopBottom()==0) ? myfpga = "bot" : myfpga = "top";

  sprintf(myline,"%1d %6d %6d %6d %6d %6s %7s",0,myside_,hcalcalibid.ieta(),hcalcalibid.iphi(),mydphi_,mydet.c_str(),rbxid.rbx().c_str());
  sprintf(myline+strlen(myline),"%8d %6d %6d %6d %4d %5s",mywedge_,rbxid.rmFiber(),hcaleid.fiberChanId(),hcaleid.readoutVMECrateId(),hcaleid.htrSlot(),myfpga.c_str());
  sprintf(myline+strlen(myline),"%8d %7d %6d %4d %6d %8d %9s\n",hcaleid.fiberIndex(),mydcc_sl,hcaleid.spigot(),mydcc,hcaleid.dccid()+700, hcalcalibid.cboxChannel(), mycalibsubdet_.c_str());

  return myline;
}


// class ZDCLogicalMapEntry 

ZDCLogicalMapEntry::ZDCLogicalMapEntry(
          int in_fi_ch, int in_htr_fi, int in_spigot, int in_fed, int in_cr, int in_htr, std::string in_s_fpga,
          std::string in_s_det, int in_sid, int in_dep, 
          int in_x, int in_y, int in_dx, int in_det_ch, int in_cab, int in_rm, int in_qie, 
          int in_adc, int in_rm_fi
  )
{
  int mytopbot;
  (in_s_fpga=="bot") ? mytopbot = 0 : mytopbot = 1;

  myx_      = in_x;
  myy_      = in_y;
  mydx_     = in_dx;
  mycable_  = in_cab;
  myrm_     = in_rm;
  myqie_    = in_qie;
  myadc_    = in_adc;
  myrm_fi_  = in_rm_fi;

  // create the hcal electronics id
  HcalElectronicsId heid( in_fi_ch, in_htr_fi, in_spigot, in_fed - 700 );
  heid.setHTR( in_cr, in_htr, mytopbot );

  //create the hcal det id
  bool myzdccheck;
  HcalZDCDetId::Section myzdcsec;
  if (in_s_det=="ZDC_EM") myzdcsec = HcalZDCDetId::EM;
  else if (in_s_det=="ZDC_HAD") myzdcsec = HcalZDCDetId::HAD;
  else if (in_s_det=="ZDC_LUM") myzdcsec = HcalZDCDetId::LUM;
  else myzdcsec = HcalZDCDetId::Unknown;
 
  (in_sid > 0) ? myzdccheck = true : myzdccheck = false;
  HcalZDCDetId hzdcdid( myzdcsec, myzdccheck, in_det_ch );

  // store the different ids
  hcalEID_      = heid.rawId();
  hcalZDCDetID_ = hzdcdid.rawId();

}
  
char* ZDCLogicalMapEntry::printLMapLine() {

  static char myline[512];

  HcalElectronicsId hcaleid(hcalEID_);
  HcalZDCDetId hcalzdcid(hcalZDCDetID_);
  HcalGenericDetId hcalgenid(hcalZDCDetID_);

  int mydcc_sl = -1;
  int mydcc    = -1;
  if ((hcaleid.dccid()%2)==1) {
    mydcc_sl = 19;
    mydcc    = 2;
  } 
  else {
    mydcc_sl = 9;
    mydcc    = 1;
  } 

  string myfpga ="";
  string mydet ="ZDC_";
  HcalZDCDetId::Section myzdcsec = hcalzdcid.section();

  if (myzdcsec==0) mydet += "Unknown";
  else if (myzdcsec==1) mydet += "EM";
  else if (myzdcsec==2) mydet += "HAD";
  else mydet += "LUM";

  (hcaleid.htrTopBottom()==0) ? myfpga = "bot" : myfpga = "top";

  sprintf(myline,"%1d %5d %2d %2d %3d %6d %7s %7d",0,hcalzdcid.zside(),myx_,myy_,mydx_,hcalzdcid.depth(),mydet.c_str(),hcalzdcid.channel());
  sprintf(myline+strlen(myline),"%7d %3d %4d %4d %6d %6d %6d",mycable_,myrm_,myqie_,myadc_,myrm_fi_,hcaleid.fiberChanId(),hcaleid.readoutVMECrateId());
  sprintf(myline+strlen(myline),"%5d %5s %7d %7d %6d %4d %6d\n",hcaleid.htrSlot(),myfpga.c_str(),hcaleid.fiberIndex(),mydcc_sl,hcaleid.spigot(),mydcc,hcaleid.dccid()+700);

  return myline;
}


// class HTLogicalMapEntry 

HTLogicalMapEntry::HTLogicalMapEntry(
             int in_et, int in_ph,
             int in_sid, int in_dph, int in_dep, std::string in_s_chDet, int in_wed, int in_cr, int in_htr, int in_tb,
             int in_spig, int in_slb, std::string in_s_slb, std::string in_s_slb2, int in_ndat,
             std::string in_s_slnam, int in_rctcra, int in_rctcar, int in_rctcon, std::string in_s_rctnam, int in_fed
  )
{
  myside_   = in_sid;
  mydphi_   = in_dph;
  mydepth_  = in_dep;
  mywedge_  = in_wed;
  myrctcra_ = in_rctcra;
  myrctcar_ = in_rctcar;
  myrctcon_ = in_rctcon;
  
  // string data members
  mydet_    = in_s_chDet;
  myslbin_  = in_s_slb;
  myslbin2_ = in_s_slb2;
  myslnam_  = in_s_slnam;
  myrctnam_ = in_s_rctnam;

  // necessary since LMap code makes top = 0, bottom = 1, but det ids have top = 1, bottom = 0
  int top = 1;
  in_tb == 1 ? top = 0 : top = 1;
  //create an hcal electronics id for the trigger tower, idea copied from CalibCalorimetry/HcalAlgos/src/HcalDBASCIIIO.cc
  HcalElectronicsId hteid( in_ndat, in_slb, in_spig, in_fed - 700, in_cr, in_htr, top );
  //HcalElectronicsId hteid( slbCh, slb, spigot, dcc, crate, slot, top );

  //create the hcal trigger tower det id
  HcalTrigTowerDetId htrigdid( in_et, in_ph );

  // store the different ids
  hcalTrigEID_   = hteid.rawId();
  hcalTrigDetID_ = htrigdid.rawId();

}

char* HTLogicalMapEntry::printLMapLine() {
  static char myline[512];
  HcalElectronicsId hcaltrigeid(hcalTrigEID_);
  HcalTrigTowerDetId hcaltrigid(hcalTrigDetID_);
  HcalGenericDetId hcalgenid(hcalTrigDetID_);

  int mydcc_sl = 0;
  int mydcc    = 0;
  if ((hcaltrigeid.dccid()%2)==1) {
    mydcc_sl = 19;
    mydcc    = 2;
  } 
  else {
    mydcc_sl = 9;
    mydcc    = 1;
  } 

  string myfpga ="";
  ( hcaltrigeid.htrTopBottom()==0 ) ? myfpga = "bot" : myfpga = "top";

  sprintf(myline,"%1d %5d %4d %4d %5d %6d %4s %7d %6d ",0,myside_,hcaltrigid.ieta(),hcaltrigid.iphi(),mydphi_,mydepth_,mydet_.c_str(),mywedge_,hcaltrigeid.readoutVMECrateId());
  sprintf(myline+strlen(myline),"%4d %5s %7d %6d %4d %4d %6s %7s %5d ",hcaltrigeid.htrSlot(),myfpga.c_str(),mydcc_sl,hcaltrigeid.spigot(),mydcc,hcaltrigeid.slbSiteNumber(),myslbin_.c_str(),myslbin2_.c_str(),hcaltrigeid.slbChannelIndex());
  sprintf(myline+strlen(myline),"%13s %7d %7d %7d %17s %6d\n",myslnam_.c_str(),myrctcra_,myrctcar_,myrctcon_,myrctnam_.c_str(),hcaltrigeid.dccid()+700);

  return myline;
}
