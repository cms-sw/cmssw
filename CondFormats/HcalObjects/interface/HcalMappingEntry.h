#ifndef HcalMappingEntry_h
#define HcalMappingEntry_h

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/HcalDetId/interface/HcalFrontEndId.h"


/*****************************Classes****************************/

class HBHEHFLogicalMapEntry {

  /****
   *  Class to hold the L-Map entries corresponding to HB, HE and HF
   ****/

 public:
         
  // constructor from map information
  HBHEHFLogicalMapEntry( ) {}
  HBHEHFLogicalMapEntry( 
			int in_fi_ch, int in_htr_fi, int in_spig, int in_fed, int in_cr, int in_htr, std::string in_s_fpga,
			std::string in_s_det, int in_sid, int in_et, int in_ph, int in_dep,
			int in_dph, int in_wed, int in_rm, int in_rm_fi, int in_pix, int in_qie, int in_adc,
			int in_slb, int in_rctcra, int in_rctcar, int in_rctcon,
			std::string in_s_rbx, std::string in_s_slb, std::string in_s_slb2, std::string in_s_slnam, std::string in_s_rctnam
			);
  
  ~HBHEHFLogicalMapEntry() {}
  
  char* printLMapLine();
  uint32_t getLinearIndex() const {return HcalElectronicsId(hcalEID_).linearIndex();}
  const HcalElectronicsId getHcalElectronicsId() const {return HcalElectronicsId(hcalEID_);}
  const DetId getDetId() const {return DetId(hcalDetID_);}
  const HcalFrontEndId getHcalFrontEndId() const {return HcalFrontEndId(hcalFrontEndID_);}
  
 private:
  
  uint32_t hcalEID_;
  uint32_t hcalDetID_;
  uint32_t hcalFrontEndID_;
  
  // stored input data members
  int mydphi_, mywedge_, myslb_, myrctcra_, myrctcar_, myrctcon_;
  
  std::string myslbin_, myslbin2_, myslnam_, myrctnam_;
  
};

class HOHXLogicalMapEntry {

  /****
   *  Class to hold the L-Map entries corresponding to HO and HX
   ****/

 public:
         
  HOHXLogicalMapEntry( ) {}
  HOHXLogicalMapEntry( 
		      int in_fi_ch, int in_htr_fi, int in_spig, int in_fed, int in_cr, int in_htr, std::string in_s_fpga,
		      std::string in_s_det, int in_sid, int in_et, int in_ph, int in_dep,
		      int in_dph, int in_sec, int in_rm, int in_rm_fi, int in_pix, int in_qie, int in_adc,
		      std::string in_s_rbx, std::string in_s_let 
		      );

  ~HOHXLogicalMapEntry() {}

  char* printLMapLine();
  const uint32_t getLinearIndex() const {return HcalElectronicsId(hcalEID_).linearIndex();}
  const HcalElectronicsId getHcalElectronicsId() const {return HcalElectronicsId(hcalEID_);}
  const DetId getDetId() const {return DetId(hcalDetID_);}
  const HcalFrontEndId getHcalFrontEndId() const {return HcalFrontEndId(hcalFrontEndID_);}

 private:
          
  uint32_t hcalEID_;
  uint32_t hcalDetID_;
  uint32_t hcalFrontEndID_;

  // stored input data members
  int mydphi_, mysector_;
  std::string myletter_;

};


class CALIBLogicalMapEntry {

  /****
   *  Class to hold the L-Map entries corresponding to the calibration channels
   ****/


 public:
         
  CALIBLogicalMapEntry( ) {}
  CALIBLogicalMapEntry( 
		       int in_fi_ch, int in_htr_fi, int in_spig, int in_fed,  int in_cr, int in_htr, std::string in_s_fpga,  
		       std::string in_s_det, int in_et, int in_ph, int in_ch_ty, 
		       int in_sid, int in_dph, std::string in_s_rbx, int in_wed, int in_rm_fi,
		       std::string in_s_subdet
		       );
  
  ~CALIBLogicalMapEntry() {}
  
  char* printLMapLine();
  const uint32_t getLinearIndex() const {return HcalElectronicsId(hcalEID_).linearIndex();}
  const HcalElectronicsId getHcalElectronicsId() const {return HcalElectronicsId(hcalEID_);}
  const DetId getDetId() const {return DetId(hcalCalibDetID_);}
  const HcalFrontEndId getHcalFrontEndId() const {return HcalFrontEndId(hcalFrontEndID_);}

 private:
  
  uint32_t hcalEID_;
  uint32_t hcalCalibDetID_;
  uint32_t hcalFrontEndID_;

  // input data members
  int myside_, mydphi_, mywedge_;
  std::string mycalibsubdet_;
  
};


class ZDCLogicalMapEntry {
  
  /****
   *  Class to hold the L-Map entries corresponding to ZDC
   ****/

 public:

  ZDCLogicalMapEntry( ) {}
  ZDCLogicalMapEntry(
		     int in_fi_ch, int in_htr_fi, int in_spigot, int in_fed, int in_cr, int in_htr, std::string in_s_fpga,
		     std::string in_s_det, int in_sid, int in_dep, 
		     int in_x, int in_y, int in_dx, int in_det_ch, int in_cab, int in_rm, int in_qie, 
		     int in_adc, int in_rm_fi
		     );

  ~ZDCLogicalMapEntry() { }

  char* printLMapLine();
  const uint32_t getLinearIndex() const {return HcalElectronicsId(hcalEID_).linearIndex();}
  const HcalElectronicsId getHcalElectronicsId() const {return HcalElectronicsId(hcalEID_);}
  const DetId getDetId() const {return DetId(hcalZDCDetID_);}
    
 private:

  uint32_t hcalEID_;
  uint32_t hcalZDCDetID_;

  // input data members
  int myx_, myy_, mydx_, mydet_ch_, mycable_, myrm_, myqie_, myadc_, myrm_fi_;

};


class HTLogicalMapEntry {

  /****
   *  Class to hold the L-Map entries corresponding to the Trigger channels
   ****/

 public:

  HTLogicalMapEntry( ) {}
  HTLogicalMapEntry(
		    int in_et, int in_ph,
		    int in_sid, int in_dph, int in_dep, std::string in_s_chDet, int in_wed, int in_cr, int in_htr, int in_tb,
		    int in_spig, int in_slb, std::string in_s_slb, std::string in_s_slb2, int in_ndat,
		    std::string in_s_slnam, int in_rctcra, int in_rctcar, int in_rctcon, std::string in_s_rctnam, int in_fed
		    );

  ~HTLogicalMapEntry() {}

  char* printLMapLine();
  const uint32_t getLinearIndex() const {return HcalElectronicsId(hcalTrigEID_).linearIndex();}
  const HcalElectronicsId getHcalTrigElectronicsId() const {return HcalElectronicsId(hcalTrigEID_);}
  const DetId getDetId() const {return DetId(hcalTrigDetID_);}

 private:

  // no meaningful electronics id for the trigger towers, but it is possible to create one that stores various data members
  uint32_t hcalTrigEID_;
  uint32_t hcalTrigDetID_;

  // input data members
  int myside_, mydphi_, mydepth_, mywedge_;
  //int myspigot_, myslb_, myndat_,  mycrate_, myhtr_, mytb_, myfedid_;
  int myrctcra_, myrctcar_, myrctcon_;
  // string data members
  std::string mydet_, myslbin_, myslbin2_, myslnam_, myrctnam_;

};

/***************/

#endif
