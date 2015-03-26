#ifndef HcalMappingEntry_h
#define HcalMappingEntry_h

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/HcalDetId/interface/HcalFrontEndId.h"

#include <string>

////special for xml generation
//#include "CalibCalorimetry/HcalTPGAlgos/interface/HCALLMAPXMLDOMBlock.h"
//#include "CalibCalorimetry/HcalTPGAlgos/interface/HCALLMAPXMLProcessor.h"
////#include "CondFormats/HcalObjects/interface/HCALLMAPXMLDOMBlock.h"
////#include "CondFormats/HcalObjects/interface/HCALLMAPXMLProcessor.h"

//##############################Classes##############################

class HBHEHFLogicalMapEntry 
{
  //####
  //  Class to hold the L-Map entries corresponding to HB, HE and HF
  //####
 public:
         
  // constructor from map information
  HBHEHFLogicalMapEntry( ) {}
  HBHEHFLogicalMapEntry(int in_fi_ch, int in_htr_fi, int in_spig, int in_fed, int in_cr, int in_htr, std::string in_s_fpga,
			std::string in_s_det, int in_sid, int in_et, int in_ph, int in_dep,
			int in_dph, int in_wed, int in_rm, int in_rm_fi, int in_pix, int in_qie, int in_adc,
			int in_slb, int in_rctcra, int in_rctcar, int in_rctcon,
			std::string in_s_rbx, std::string in_s_slb, std::string in_s_slb2, std::string in_s_slnam, std::string in_s_rctnam,
                        int in_qieid, int in_uhtr_cr, int in_uhtr, std::string in_uhtr_fpga, int in_uhtr_dcc, int in_uhtr_spigot, int in_uhtr_fi, int in_uhtr_fedid);
    
  ~HBHEHFLogicalMapEntry() {}
  
  char* printLMapLine();
  char* printLMapLine_uhtr();
  //HCALLMAPXMLProcessor::LMapRowHBEF generateXMLRow();
  uint32_t getLinearIndex() const {return HcalElectronicsId(hcalEID_).linearIndex();}
  const HcalElectronicsId getHcalElectronicsId() const {return HcalElectronicsId(hcalEID_);}
  const DetId getDetId() const {return DetId(hcalDetID_);}
  const HcalFrontEndId getHcalFrontEndId() const {return HcalFrontEndId(hcalFrontEndID_);}
  
 protected:
  uint32_t hcalEID_;
  uint32_t hcalDetID_;
  uint32_t hcalFrontEndID_;
  
 public:
  //stored input data members
  //shared by htr and uhtr system;
  int mydphi_, mywedge_, myslb_, myrctcra_, myrctcar_, myrctcon_;
  int myqieid_; 
  std::string myslbin_, myslbin2_, myslnam_, myrctnam_;
 
  //uhtr variables
  //the vraiables need to be changed in uhtr
  int myuhtr_crate_, myuhtr_, myuhtr_dcc_, myuhtr_spigot_, myuhtr_htr_fi_, myuhtr_fedid_;
  std::string myuhtr_fpga_;
  
  //the variables keeps all the same in uhtr
  //hcal detector id 
  std::string mydet_;
  int mysid_, myet_, myph_, mydep_;
  //hcal front end id
  std::string myrbx_;  
  int myrm_, myrm_fi_, mypix_, myqie_, myadc_, myfi_ch_;

  //the i variable in uhtr emap,just a copy of hcalEID_ for now, added by hua.wei@cern.ch
  uint32_t hcalDetID_uhtr_;
};


class HOHXLogicalMapEntry 
{
  //####
  //  Class to hold the L-Map entries corresponding to HO and HX
  //####
 public:
         
  HOHXLogicalMapEntry( ) {}
  HOHXLogicalMapEntry(
                      int in_fi_ch, int in_htr_fi, int in_spig, int in_fed, int in_cr, double in_block_coupler, int in_htr, std::string in_s_fpga,
		      std::string in_s_det, int in_sid, int in_et, int in_ph, int in_dep,
		      int in_dph, int in_sec, int in_rm, int in_rm_fi, int in_pix, int in_qie, int in_adc,
		      std::string in_s_rbx, std::string in_s_let, 
                      int in_qieid
                     );

  ~HOHXLogicalMapEntry() {}

  char* printLMapLine();
  //HCALLMAPXMLProcessor::LMapRowHO generateXMLRow();
  const uint32_t getLinearIndex() const {return HcalElectronicsId(hcalEID_).linearIndex();}
  const HcalElectronicsId getHcalElectronicsId() const {return HcalElectronicsId(hcalEID_);}
  const DetId getDetId() const {return DetId(hcalDetID_);}
  const HcalFrontEndId getHcalFrontEndId() const {return HcalFrontEndId(hcalFrontEndID_);}

 protected:
          
  uint32_t hcalEID_;
  uint32_t hcalDetID_;
  uint32_t hcalFrontEndID_;

 public:
  //stored input data members
  int mydphi_, mysector_;
  std::string myletter_;
  int myqieid_;
  double myblock_coupler_;
};


class CALIBLogicalMapEntry 
{
  //####
  //  Class to hold the L-Map entries corresponding to the calibration channels
  //####

 public:
         
  CALIBLogicalMapEntry( ) {}
  CALIBLogicalMapEntry(int in_fi_ch, int in_htr_fi, int in_spig, int in_fed,  int in_cr, int in_htr, std::string in_s_fpga,  
		       std::string in_s_det, int in_et, int in_ph, int in_ch_ty, 
		       int in_sid, int in_dph, std::string in_s_rbx, int in_wed, int in_rm_fi,
		       std::string in_s_subdet);
  
  ~CALIBLogicalMapEntry() {}
  
  char* printLMapLine();
  //HCALLMAPXMLProcessor::LMapRowCALIB generateXMLRow();
  const uint32_t getLinearIndex() const {return HcalElectronicsId(hcalEID_).linearIndex();}
  const HcalElectronicsId getHcalElectronicsId() const {return HcalElectronicsId(hcalEID_);}
  const DetId getDetId() const {return DetId(hcalCalibDetID_);}
  const HcalFrontEndId getHcalFrontEndId() const {return HcalFrontEndId(hcalFrontEndID_);}

 protected:
  
  uint32_t hcalEID_;
  uint32_t hcalCalibDetID_;
  uint32_t hcalFrontEndID_;

 public:
  //input data members
  int myside_, mydphi_, mywedge_;
  std::string mycalibsubdet_;
  
};


class ZDCLogicalMapEntry 
{
  //####
  //  Class to hold the L-Map entries corresponding to ZDC
  //####
 public:
  ZDCLogicalMapEntry( ) {}
  ZDCLogicalMapEntry(int in_fi_ch, int in_htr_fi, int in_spigot, int in_fed, int in_cr, int in_htr, std::string in_s_fpga,
		     std::string in_s_det, int in_sid, int in_dep, 
		     int in_x, int in_y, int in_dx, int in_det_ch, int in_cab, int in_rm, int in_qie, 
		     int in_adc, int in_rm_fi);

  ~ZDCLogicalMapEntry() { }

  char* printLMapLine();
  //HCALLMAPXMLProcessor::LMapRowZDC generateXMLRow();
  const uint32_t getLinearIndex() const {return HcalElectronicsId(hcalEID_).linearIndex();}
  const HcalElectronicsId getHcalElectronicsId() const {return HcalElectronicsId(hcalEID_);}
  const DetId getDetId() const {return DetId(hcalZDCDetID_);}
    
 protected:

  uint32_t hcalEID_;
  uint32_t hcalZDCDetID_;

 public:
  //input data members
  int myx_, myy_, mydx_, mydet_ch_, mycable_, myrm_, myqie_, myadc_, myrm_fi_;
};


class HTLogicalMapEntry 
{

  //####
  //   Class to hold the L-Map entries corresponding to the Trigger channels
  //####
 public:
  HTLogicalMapEntry( ) {}
  HTLogicalMapEntry(
		    int in_et, int in_ph,
		    int in_sid, int in_dph, int in_dep, std::string in_s_chDet, int in_wed, int in_cr, int in_htr, int in_tb,
		    int in_spig, int in_slb, std::string in_s_slb, std::string in_s_slb2, int in_ndat,
		    std::string in_s_slnam, int in_rctcra, int in_rctcar, int in_rctcon, std::string in_s_rctnam, int in_fed,
                    int in_qieid
		    );

  ~HTLogicalMapEntry() {}

  char* printLMapLine();
  //HCALLMAPXMLProcessor::LMapRowHT generateXMLRow();
  const uint32_t getLinearIndex() const {return HcalElectronicsId(hcalTrigEID_).linearIndex();}
  const HcalElectronicsId getHcalTrigElectronicsId() const {return HcalElectronicsId(hcalTrigEID_);}
  const DetId getDetId() const {return DetId(hcalTrigDetID_);}

 private:

  //no meaningful electronics id for the trigger towers, but it is possible to create one that stores various data members
  uint32_t hcalTrigEID_;
  uint32_t hcalTrigDetID_;

  public:
  //input data members
  int myside_, mydphi_, mydepth_, mywedge_;
  //int myspigot_, myslb_, myndat_,  mycrate_, myhtr_, mytb_, myfedid_;
  int myrctcra_, myrctcar_, myrctcon_;
  //string data members
  std::string mydet_, myslbin_, myslbin2_, myslnam_, myrctnam_;

  int myqieid_;
};

//#################################//
//The following is the structs/classes defined for QIE Id variables, added by hua.wei@cern.ch
struct HBHEHFQIEInfo
{
  int subdet;//HB,0;HE,1;HF,2
  int side;
  int rbxno;
  int rm;
  int qie;

  int qieid;
  //float slopes[16];
  //float offsets[16];  
};

struct HOR0QIEInfo
{
  int rbxno;
  int rm;
  int qie;

  int qieid;
  //float slopes[16];
  //float offsets[16]; 
};

struct HOR12QIEInfo
{
  int side;
  int ring;
  int rbxno;
  int rm;
  int qie;

  int qieid;
  //float slopes[16];
  //float offsets[16]; 
};

struct QIECardNormal
{
  int qieid;
  int ch;
  int cap;
  int rng;
  float offset;
  float slope;
  float fcs[32];
};

struct QIECardCalib
{
  int qieid;
  int ch;
  float offset;
  float slope;
  float bins[32];
};

struct QIEMap
{
  char det[2];
  int eta;
  int phi;
  int depth;
  int qieid;
  int qie_ch;
};

struct OfflineDB
{
  int qieid;
  int qie_ch;
  int eta;
  int phi;
  int depth;
  char det[2];
  float offsets[16];
  float slopes[16];
};

//########################################//
#endif
