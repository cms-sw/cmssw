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

HBHEHFLogicalMapEntry::HBHEHFLogicalMapEntry(int in_fi_ch, int in_htr_fi, int in_spig, int in_fed, int in_cr, int in_htr, std::string in_s_fpga,
                                             std::string in_s_det, int in_sid, int in_et, int in_ph, int in_dep,
                                             int in_dph, int in_wed, int in_rm, int in_rm_fi, int in_pix, int in_qie, int in_adc,
                                             int in_slb, int in_rctcra, int in_rctcar, int in_rctcon,
                                             std::string in_s_rbx, std::string in_s_slb, std::string in_s_slb2, std::string in_s_slnam, std::string in_s_rctnam,
                                             int in_qieid,int in_uhtr_cr, int in_uhtr, std::string in_uhtr_fpga, int in_uhtr_dcc, int in_uhtr_spigot, int in_uhtr_fi, int in_uhtr_fedid)
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
  
  mydet_=in_s_det;
  mysid_=in_sid; 
  myet_ =in_et; 
  myph_ =in_ph; 
  mydep_=in_dep;

  HcalDetId hdid( mysubdet, in_sid*in_et, in_ph, in_dep );

  myrbx_  =in_s_rbx;  
  myrm_   =in_rm; 
  myrm_fi_=in_rm_fi; 
  mypix_  =in_pix; 
  myqie_  =in_qie; 
  myadc_  =in_adc; 
  myfi_ch_=in_fi_ch;

  HcalFrontEndId hrbx( in_s_rbx, in_rm, in_pix, in_rm_fi, in_fi_ch, in_qie, in_adc );

  // store the different ids
  hcalEID_        = heid.rawId(); //backend electronics, determined by fi_ch, htr_fiber. spigot, and fed
  hcalDetID_      = hdid.rawId(); //detector, determined by subdetector, eta, phi, depth
  hcalFrontEndID_ = hrbx.rawId(); //frontend, determined by rbx, rm, pixel, rm_fiber qie and adc

  //QIEID inpiut
  myqieid_ = in_qieid;

  //micro TCA variables
  myuhtr_crate_ = in_uhtr_cr;
  myuhtr_       = in_uhtr;
  myuhtr_dcc_   = in_uhtr_dcc;
  myuhtr_spigot_= in_uhtr_spigot;
  myuhtr_htr_fi_= in_uhtr_fi;
  myuhtr_fedid_ = in_uhtr_fedid;

  myuhtr_fpga_  = in_uhtr_fpga;

  //for hcal uhtr emap, added by hua.wei@cern.ch
  hcalDetID_uhtr_ = hcalDetID_;
}

char* HBHEHFLogicalMapEntry::printLMapLine() 
{
  static char myline[512];

  HcalElectronicsId hcaleid(hcalEID_);
  HcalDetId         hcaldid(hcalDetID_);
  HcalGenericDetId  hcalgenid(hcalDetID_);
  HcalFrontEndId    rbxid(hcalFrontEndID_);

  int mydcc_sl = 0;
  int mydcc    = 0;
  if ((hcaleid.dccid()%2)==1) 
  {
    mydcc_sl = 20;
    mydcc    = 2;
  } 
  else 
  {
    mydcc_sl = 10;
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
  sprintf(myline+strlen(myline),"%8s %15s    %6d %6d %6d %20s    %6d",myslbin2_.c_str(),myslnam_.c_str(),myrctcra_,myrctcar_,myrctcon_,myrctnam_.c_str(),hcaleid.dccid()+700);
  sprintf(myline+strlen(myline)," %6d\n",myqieid_);

  return myline;
}

char* HBHEHFLogicalMapEntry::printLMapLine_uhtr() 
{
  static char myline[512];

  int mydcc_sl_ = 0;
  
  sprintf(myline,"%1d %6d %6d %6d %6d %6d %6s %7s %6d %6d %6d",0,mysid_,myet_,myph_,mydphi_,mydep_,mydet_.c_str(),myrbx_.c_str(),mywedge_,myrm_,mypix_);
  sprintf(myline+strlen(myline),"%6d %6d %6d %6d %6d %6d %6s",myqie_,myadc_,myrm_fi_,myfi_ch_,myuhtr_crate_,myuhtr_,myuhtr_fpga_.c_str());
  sprintf(myline+strlen(myline),"%8d %7d %6d %6d %6d %6s",myuhtr_htr_fi_,mydcc_sl_,myuhtr_spigot_,myuhtr_dcc_,myslb_,myslbin_.c_str());
  sprintf(myline+strlen(myline),"%8s %15s    %6d %6d %6d %20s    %6d",myslbin2_.c_str(),myslnam_.c_str(),myrctcra_,myrctcar_,myrctcon_,myrctnam_.c_str(),myuhtr_fedid_);
  sprintf(myline+strlen(myline)," %6d\n",myqieid_);

  return myline;
}


//HCALLMAPXMLProcessor::LMapRowHBEF HBHEHFLogicalMapEntry::generateXMLRow(){
//
//  HCALLMAPXMLProcessor::LMapRowHBEF hbefRow;
//
//  HcalElectronicsId hcaleid(hcalEID_);
//  HcalDetId hcaldid(hcalDetID_);
//  HcalGenericDetId hcalgenid(hcalDetID_);
//  HcalFrontEndId rbxid(hcalFrontEndID_);
//
//  int mydcc_sl = 0;
//  int mydcc    = 0;
//  if ((hcaleid.dccid()%2)==1) {
//    mydcc_sl = 20;
//    mydcc    = 2;
//  } 
//  else {
//    mydcc_sl = 10;
//    mydcc    = 1;
//  } 
//
//  string myfpga = "";
//  string mydet  = "";
//  HcalSubdetector mysubdet = hcaldid.subdet();
//  (mysubdet==1) ? mydet = "HB" : 
//              ((mysubdet==2) ? mydet = "HE" :
//              ((mysubdet==3) ? mydet = "HO" : 
//              ((mysubdet==4) ? mydet = "HF" : 
//              mydet = "invalid")));
//
//  ( hcaleid.htrTopBottom()==0 ) ? myfpga = "bot" : myfpga = "top";
//
//  hbefRow . side   = hcaldid.zside();
//  hbefRow . eta    = hcaldid.ietaAbs();
//  hbefRow . phi    = hcaldid.iphi();
//  hbefRow . dphi   = mydphi_;
//  hbefRow . depth  = hcaldid.depth();
//  hbefRow . det    = mydet.c_str();
//  hbefRow . rbx    = rbxid.rbx().c_str();
//  hbefRow . wedge  = mywedge_;
//  hbefRow . rm     = rbxid.rm();
//  hbefRow . pixel  = rbxid.pixel();
//  hbefRow . qie    = rbxid.qieCard();
//  hbefRow . adc    = rbxid.adc();
//  hbefRow . rm_fi  = rbxid.rmFiber();
//  hbefRow . fi_ch  = hcaleid.fiberChanId();
//  hbefRow . crate  = hcaleid.readoutVMECrateId();
//  hbefRow . htr    = hcaleid.htrSlot();
//  hbefRow . fpga   = myfpga.c_str();
//  hbefRow . htr_fi = hcaleid.fiberIndex();
//  hbefRow . dcc_sl = mydcc_sl;
//  hbefRow . spigo  = hcaleid.spigot();
//  hbefRow . dcc    = mydcc;
//  hbefRow . slb    = myslb_;
//  hbefRow . slbin  = myslbin_.c_str();
//  hbefRow . slbin2 = myslbin2_.c_str();
//  hbefRow . slnam  = myslnam_.c_str();
//  hbefRow . rctcra = myrctcra_;
//  hbefRow . rctcar = myrctcar_;
//  hbefRow . rctcon = myrctcon_;
//  hbefRow . rctnam = myrctnam_.c_str();
//  hbefRow . fedid  = hcaleid.dccid()+700;
//
//  return hbefRow;
//}


// class HOHXLogicalMapEntry 
HOHXLogicalMapEntry::HOHXLogicalMapEntry(int in_fi_ch, int in_htr_fi, int in_spig, int in_fed, int in_cr, double in_block_coupler, int in_htr, std::string in_s_fpga,
                                         std::string in_s_det, int in_sid, int in_et, int in_ph, int in_dep,
                                         int in_dph, int in_sec, int in_rm, int in_rm_fi, int in_pix, int in_qie, int in_adc,
                                         std::string in_s_rbx, std::string in_s_let,
                                         int in_qieid)
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

  if (in_s_det=="HO") 
  {
    //create the hcal det id in the case of regular HO channel
    HcalDetId hdid( HcalOuter, in_sid*in_et, in_ph, in_dep );
    hcalDetID_ = hdid.rawId();
  }
  else 
  {
    //create the calib det id in the case of HO cross talk channels
    HcalCalibDetId hdid( in_sid*in_et, in_ph );
    hcalDetID_ = hdid.rawId();
  }

  HcalFrontEndId hrbx( in_s_rbx, in_rm, in_pix, in_rm_fi, in_fi_ch, in_qie, in_adc );

  // store the different ids
  hcalEID_        = heid.rawId();
  hcalFrontEndID_ = hrbx.rawId();

  //QIEID input
  myqieid_ = in_qieid;

  //Patch Panel Block and coupler input
  myblock_coupler_ = in_block_coupler;
}
  
char* HOHXLogicalMapEntry::printLMapLine() 
{

  static char myline[512];

  HcalElectronicsId hcaleid(hcalEID_);
  HcalGenericDetId hcalgenid(hcalDetID_);
  HcalFrontEndId rbxid(hcalFrontEndID_);

  int mydcc_sl = 0;
  int mydcc    = 0;
  if ((hcaleid.dccid()%2)==1) 
  {
    mydcc_sl = 20;
    mydcc    = 2;
  } 
  else 
  {
    mydcc_sl = 10;
    mydcc    = 1;
  } 

  string myfpga = "";
  string mydet  = "";
  int mydepth   = 0;
  int myside    = -2;
  int myeta     = 0;
  int myphi     = -1;
  if ( hcalgenid.isHcalCalibDetId() ) 
  {
    HcalCalibDetId hcalcompid(hcalDetID_);
    mydet   = "HOX";
    mydepth = 4;
    myside  = hcalcompid.zside();
    myeta   = hcalcompid.ieta()*myside;
    myphi   = hcalcompid.iphi();
  }
  else if ( hcalgenid.isHcalDetId() ) 
  {
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

  sprintf( myline , "%1d %6d %6d %6d %6d %6d %6s %7s %6d %6d %6d" , 0 , myside , myeta , myphi , mydphi_ , mydepth , mydet.c_str() , rbxid.rbx().c_str() , mysector_,rbxid.rm() , rbxid.pixel() );
  sprintf( myline+strlen(myline) , "%6d %6d %6d %6d %8s %6d %12.2f %6d %6s" , rbxid.qieCard() , rbxid.adc() , rbxid.rmFiber() , hcaleid.fiberChanId() , myletter_.c_str() , hcaleid.readoutVMECrateId() , myblock_coupler_ , hcaleid.htrSlot() , myfpga.c_str() );
  sprintf( myline+strlen(myline) , "%8d %7d %6d %6d %6d" , hcaleid.fiberIndex() , mydcc_sl , hcaleid.spigot() , mydcc , hcaleid.dccid()+700 );
  sprintf( myline+strlen(myline) , " %6d\n" , myqieid_ );
  return myline;
}

//HCALLMAPXMLProcessor::LMapRowHO HOHXLogicalMapEntry::generateXMLRow(){
//
//  HCALLMAPXMLProcessor::LMapRowHO hoxRow;
//
//  HcalElectronicsId hcaleid(hcalEID_);
//  HcalGenericDetId hcalgenid(hcalDetID_);
//  HcalFrontEndId rbxid(hcalFrontEndID_);
//
//  int mydcc_sl = 0;
//  int mydcc    = 0;
//  if ((hcaleid.dccid()%2)==1) {
//    mydcc_sl = 20;
//    mydcc    = 2;
//  } 
//  else {
//    mydcc_sl = 10;
//    mydcc    = 1;
//  } 
//
//  string myfpga = "";
//  string mydet  = "";
//  int mydepth   = 0;
//  int myside    = -2;
//  int myeta     = 0;
//  int myphi     = -1;
//  if ( hcalgenid.isHcalCalibDetId() ) {
//    HcalCalibDetId hcalcompid(hcalDetID_);
//    mydet   = "HOX";
//    mydepth = 4;
//    myside  = hcalcompid.zside();
//    myeta   = hcalcompid.ieta()*myside;
//    myphi   = hcalcompid.iphi();
//  }
//  else if ( hcalgenid.isHcalDetId() ) {
//    HcalDetId hcalcompid(hcalDetID_);
//    HcalSubdetector mysubdet = hcalcompid.subdet();
//    (mysubdet==HcalBarrel) ? mydet = "HB" : 
//                ((mysubdet==HcalEndcap) ? mydet = "HE" :
//                ((mysubdet==HcalOuter) ? mydet = "HO" : 
//                ((mysubdet==HcalForward) ? mydet = "HF" : 
//                mydet = "invalid")));
//    mydepth = hcalcompid.depth();
//    myside  = hcalcompid.zside();  
//    myeta   = hcalcompid.ietaAbs();
//    myphi   = hcalcompid.iphi();
//  }
//
//  ( hcaleid.htrTopBottom()==0 ) ? myfpga = "bot" : myfpga = "top";
//
//  hoxRow . sideO     = myside;
//  hoxRow . etaO      = myeta;
//  hoxRow . phiO      = myphi;
//  hoxRow . dphiO     = mydphi_;
//  hoxRow . depthO    = mydepth;
//  hoxRow . detO      = mydet.c_str();
//  hoxRow . rbxO      = rbxid.rbx().c_str();
//  hoxRow . sectorO   = mysector_;
//  hoxRow . rmO       = rbxid.rm();
//  hoxRow . pixelO    = rbxid.pixel();
//  hoxRow . qieO      = rbxid.qieCard();
//  hoxRow . adcO      = rbxid.adc();
//  hoxRow . rm_fiO    = rbxid.rmFiber();
//  hoxRow . fi_chO    = hcaleid.fiberChanId();
//  hoxRow . let_codeO = myletter_.c_str();
//  hoxRow . crateO    = hcaleid.readoutVMECrateId();
//  hoxRow . htrO      = hcaleid.htrSlot();
//  hoxRow . fpgaO     = myfpga.c_str();
//  hoxRow . htr_fiO   = hcaleid.fiberIndex();
//  hoxRow . dcc_slO   = mydcc_sl;
//  hoxRow . spigoO    = hcaleid.spigot();
//  hoxRow . dccO      = mydcc;
//  hoxRow . fedidO    = hcaleid.dccid()+700;
//
//  return hoxRow;
//}


// class CalibLogicalMapEntry 

CALIBLogicalMapEntry::CALIBLogicalMapEntry(int in_fi_ch, int in_htr_fi, int in_spig, int in_fed,  int in_cr, int in_htr, std::string in_s_fpga,  
                                           std::string in_s_det, int in_et, int in_ph, int in_ch_ty, 
                                           int in_sid, int in_dph, std::string in_s_rbx, int in_wed, int in_rm_fi,
                                           std::string in_s_subdet)
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
  //CM RM in HF is 4 rather than 5
  if (in_s_det=="HF")
    in_rm  = 4;
  else
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

char* CALIBLogicalMapEntry::printLMapLine() 
{

  static char myline[512];

  HcalElectronicsId hcaleid(hcalEID_);
  HcalCalibDetId    hcalcalibid(hcalCalibDetID_);
  HcalGenericDetId  hcalgenid(hcalCalibDetID_);
  HcalFrontEndId    rbxid(hcalFrontEndID_);

  int mydcc_sl = 0;
  int mydcc    = 0;
  if ((hcaleid.dccid()%2)==1) 
  {
    mydcc_sl = 20;
    mydcc    = 2;
  } 
  else 
  {
    mydcc_sl = 10;
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
  sprintf(myline+strlen(myline),"%8d %6d %6d %6d %6d %4d %5s",mywedge_,rbxid.rm(),rbxid.rmFiber(),hcaleid.fiberChanId(),hcaleid.readoutVMECrateId(),hcaleid.htrSlot(),myfpga.c_str());
  sprintf(myline+strlen(myline),"%8d %7d %6d %4d %6d %8d %9s\n",hcaleid.fiberIndex(),mydcc_sl,hcaleid.spigot(),mydcc,hcaleid.dccid()+700, hcalcalibid.cboxChannel(), mycalibsubdet_.c_str());

  return myline;
}

//HCALLMAPXMLProcessor::LMapRowCALIB CALIBLogicalMapEntry::generateXMLRow(){
//
//  HCALLMAPXMLProcessor::LMapRowCALIB calibRow;
//
//  HcalElectronicsId hcaleid(hcalEID_);
//  HcalCalibDetId hcalcalibid(hcalCalibDetID_);
//  HcalGenericDetId hcalgenid(hcalCalibDetID_);
//  HcalFrontEndId rbxid(hcalFrontEndID_);
//
//  int mydcc_sl = 0;
//  int mydcc    = 0;
//  if ((hcaleid.dccid()%2)==1) {
//    mydcc_sl = 20;
//    mydcc    = 2;
//  } 
//  else {
//    mydcc_sl = 10;
//    mydcc    = 1;
//  } 
//
//  string myfpga = "";
//  string mydet  = "";
//  HcalSubdetector mysubdet = hcalcalibid.hcalSubdet();
//  (mysubdet==HcalBarrel) ? mydet = "HB" : 
//              ((mysubdet==HcalEndcap)  ? mydet = "HE" :
//              ((mysubdet==HcalOuter)   ? mydet = "HO" : 
//              ((mysubdet==HcalForward) ? mydet = "HF" : 
//              mydet = "invalid")));
//  (hcaleid.htrTopBottom()==0) ? myfpga = "bot" : myfpga = "top";
//
//  calibRow . sideC     = myside_;
//  calibRow . etaC      = hcalcalibid.ieta();
//  calibRow . phiC      = hcalcalibid.iphi();
//  calibRow . dphiC     = mydphi_;
//  calibRow . detC      = mydet.c_str();
//  calibRow . rbxC      = rbxid.rbx().c_str();
//  calibRow . sectorC   = mywedge_;
//  calibRow . rmC       = rbxid.rm();
//  calibRow . rm_fiC    = rbxid.rmFiber();
//  calibRow . fi_chC    = hcaleid.fiberChanId();
//  calibRow . crateC    = hcaleid.readoutVMECrateId();
//  calibRow . htrC      = hcaleid.htrSlot();
//  calibRow . fpgaC     = myfpga.c_str();
//  calibRow . htr_fiC   = hcaleid.fiberIndex();
//  calibRow . dcc_slC   = mydcc_sl;
//  calibRow . spigoC    = hcaleid.spigot();
//  calibRow . dccC      = mydcc;
//  calibRow . fedidC    = hcaleid.dccid()+700;
//  calibRow . ch_typeC  = hcalcalibid.cboxChannel();
//
//  return calibRow;
//}


// class ZDCLogicalMapEntry 

ZDCLogicalMapEntry::ZDCLogicalMapEntry(int in_fi_ch, int in_htr_fi, int in_spigot, int in_fed, int in_cr, int in_htr, std::string in_s_fpga,
                                       std::string in_s_det, int in_sid, int in_dep, 
                                       int in_x, int in_y, int in_dx, int in_det_ch, int in_cab, int in_rm, int in_qie, 
                                       int in_adc, int in_rm_fi)
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
  
char* ZDCLogicalMapEntry::printLMapLine() 
{

  static char myline[512];

  HcalElectronicsId hcaleid(hcalEID_);
  HcalZDCDetId hcalzdcid(hcalZDCDetID_);
  HcalGenericDetId hcalgenid(hcalZDCDetID_);

  int mydcc_sl = -1;
  int mydcc    = -1;
  if ((hcaleid.dccid()%2)==1) 
  {
    mydcc_sl = 20;
    mydcc    = 2;
  } 
  else 
  {
    mydcc_sl = 10;
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

//HCALLMAPXMLProcessor::LMapRowZDC ZDCLogicalMapEntry::generateXMLRow(){
//
//  HCALLMAPXMLProcessor::LMapRowZDC zdcRow;
//
//  HcalElectronicsId hcaleid(hcalEID_);
//  HcalZDCDetId hcalzdcid(hcalZDCDetID_);
//  HcalGenericDetId hcalgenid(hcalZDCDetID_);
//
//  int mydcc_sl = -1;
//  int mydcc    = -1;
//  if ((hcaleid.dccid()%2)==1) {
//    mydcc_sl = 20;
//    mydcc    = 2;
//  } 
//  else {
//    mydcc_sl = 10;
//    mydcc    = 1;
//  } 
//
//  string myfpga ="";
//  string mydet ="ZDC_";
//  HcalZDCDetId::Section myzdcsec = hcalzdcid.section();
//
//  if (myzdcsec==0) mydet = "Unknown";
//  else if (myzdcsec==1) mydet = "EM";
//  else if (myzdcsec==2) mydet = "HAD";
//  else mydet = "LUM";
//
//  (hcaleid.htrTopBottom()==0) ? myfpga = "bot" : myfpga = "top";
//
//  zdcRow . sideZ   = hcalzdcid.zside();
//  zdcRow . xZ      = myx_;
//  zdcRow . yZ      = myy_;
//  zdcRow . dxZ     = mydx_;
//  zdcRow . depthZ  = hcalzdcid.depth();
//  zdcRow . detZ    = mydet.c_str();
//  zdcRow . det_chZ = hcalzdcid.channel();
//  zdcRow . cableZ  = mycable_;
//  zdcRow . rmZ     = myrm_;
//  zdcRow . qieZ    = myqie_;
//  zdcRow . adcZ    = myadc_;
//  zdcRow . rm_fiZ  = myrm_fi_;
//  zdcRow . fi_chZ  = hcaleid.fiberChanId();
//  zdcRow . crateZ  = hcaleid.readoutVMECrateId();
//  zdcRow . htrZ    = hcaleid.htrSlot();
//  zdcRow . fpgaZ   = myfpga.c_str();
//  zdcRow . htr_fiZ = hcaleid.fiberIndex();
//  zdcRow . dcc_slZ = mydcc_sl;
//  zdcRow . spigoZ  = hcaleid.spigot();
//  zdcRow . dccZ    = mydcc;
//  zdcRow . fedidZ  = hcaleid.dccid()+700;
//
//  return zdcRow;
//}


// class HTLogicalMapEntry 

HTLogicalMapEntry::HTLogicalMapEntry(int in_et, int in_ph,
                                     int in_sid, int in_dph, int in_dep, std::string in_s_chDet, int in_wed, int in_cr, int in_htr, int in_tb,
                                     int in_spig, int in_slb, std::string in_s_slb, std::string in_s_slb2, int in_ndat,
                                     std::string in_s_slnam, int in_rctcra, int in_rctcar, int in_rctcon, std::string in_s_rctnam, int in_fed,
                                     int in_qieid)
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

  //QIEID entry
  myqieid_ = in_qieid;

}

char* HTLogicalMapEntry::printLMapLine() 
{
  static char myline[512];
  HcalElectronicsId hcaltrigeid(hcalTrigEID_);
  HcalTrigTowerDetId hcaltrigid(hcalTrigDetID_);
  HcalGenericDetId hcalgenid(hcalTrigDetID_);

  int mydcc_sl = 0;
  int mydcc    = 0;
  if ((hcaltrigeid.dccid()%2)==1) 
  {
    mydcc_sl = 20;
    mydcc    = 2;
  } 
  else 
  {
    mydcc_sl = 10;
    mydcc    = 1;
  } 

  string myfpga ="";
  ( hcaltrigeid.htrTopBottom()==0 ) ? myfpga = "bot" : myfpga = "top";

  sprintf(myline,"%1d %5d %4d %4d %5d %6d %4s %7d %6d ",0,myside_,hcaltrigid.ieta(),hcaltrigid.iphi(),mydphi_,mydepth_,mydet_.c_str(),mywedge_,hcaltrigeid.readoutVMECrateId());
  sprintf(myline+strlen(myline),"%4d %5s %7d %6d %4d %4d %6s %7s %5d ",hcaltrigeid.htrSlot(),myfpga.c_str(),mydcc_sl,hcaltrigeid.spigot(),mydcc,hcaltrigeid.slbSiteNumber(),myslbin_.c_str(),myslbin2_.c_str(),hcaltrigeid.slbChannelIndex());
  sprintf(myline+strlen(myline),"%13s %7d %7d %7d %17s %6d\n",myslnam_.c_str(),myrctcra_,myrctcar_,myrctcon_,myrctnam_.c_str(),hcaltrigeid.dccid()+700);
  //sprintf(myline+strlen(myline)," %6d\n",myqieid_);
  return myline;
}

//HCALLMAPXMLProcessor::LMapRowHT HTLogicalMapEntry::generateXMLRow(){
//
//  HCALLMAPXMLProcessor::LMapRowHT htRow;
//
//  HcalElectronicsId hcaltrigeid(hcalTrigEID_);
//  HcalTrigTowerDetId hcaltrigid(hcalTrigDetID_);
//  HcalGenericDetId hcalgenid(hcalTrigDetID_);
//
//  int mydcc_sl = 0;
//  int mydcc    = 0;
//  if ((hcaltrigeid.dccid()%2)==1) {
//    mydcc_sl = 19;
//    mydcc    = 2;
//  } 
//  else {
//    mydcc_sl = 9;
//    mydcc    = 1;
//  } 
//
//  string myfpga ="";
//  ( hcaltrigeid.htrTopBottom()==0 ) ? myfpga = "bot" : myfpga = "top";
//
//  htRow . sideT   = myside_;
//  htRow . etaT    = hcaltrigid.ieta();
//  htRow . phiT    = hcaltrigid.iphi();
//  htRow . dphiT   = mydphi_;
//  htRow . depthT  = mydepth_;
//  htRow . detT    = mydet_.c_str();
//  htRow . wedgeT  = mywedge_;
//  htRow . crateT  = hcaltrigeid.readoutVMECrateId();
//  htRow . htrT    = hcaltrigeid.htrSlot();
//  htRow . fpgaT   = myfpga.c_str();
//  htRow . dcc_slT = mydcc_sl;
//  htRow . spigoT  = hcaltrigeid.spigot();
//  htRow . dccT    = mydcc;
//  htRow . slbT    = hcaltrigeid.slbSiteNumber();
//  htRow . slbinT  = myslbin_.c_str();
//  htRow . slbin2T = myslbin2_.c_str();
//  htRow . ndatT   = hcaltrigeid.slbChannelIndex();
//  htRow . slnamT  = myslnam_.c_str();
//  htRow . rctcraT = myrctcra_;
//  htRow . rctcarT = myrctcar_;
//  htRow . rctconT = myrctcon_;
//  htRow . rctnamT = myrctnam_.c_str();
//  htRow . fedidT  = hcaltrigeid.dccid()+700;
//
//  return htRow;
//}
