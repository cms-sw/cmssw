#ifndef DATAFORMATS_HCALDETID_HCALCALIBDETID_H
#define DATAFORMATS_HCALDETID_HCALCALIBDETID_H 1

#include <ostream>
#include "DataFormats/HcalDetId/interface/HcalOtherDetId.h"


/** \class HcalCalibDetId
  *  
  *  Contents of the HcalCalibDetId :
  *     [19:17] Calibration Category (1 => CalibUnit)
  *
  *  For CalibUnit:
  *     [12:9] Channel number/type (see below)
  *     [8:5] Detector sector (HB+,HB-,HE+,HE-,HF+,HF-,HO2-,HO1-,HOO,HO1+,HO2+)
  *     [4:0] RBX number
  *
  * $Date: 2006/06/16 16:44:04 $
  * $Revision: 1.4 $
  * \author J. Mans - Minnesota
  */
class HcalCalibDetId : public HcalOtherDetId {
public:
  /** Type identifier within calibration det ids */
  enum CalibDetType { CalibrationBox = 1 };
  /** Detector sector numbering ENUM */
  enum SectorId { HBplus=1, HBminus=2, 
		  HEplus=3, HEminus=4, 
		  HO2plus=5, HO1plus=6, HOzero=7, HO1minus=8, HO2minus=9, 
		  HFplus=10, HFminus=11 }; 

  /** Create a null det id */
  HcalCalibDetId();
  /** Create a from a raw id */
  HcalCalibDetId(uint32_t rawid);
  /** Create from a generic cell id */
  HcalCalibDetId(const DetId& id);
  HcalCalibDetId& operator=(const DetId& id);
  /** Construct a calibration box - channel detid */
  HcalCalibDetId(SectorId sector, int rbx, int channel);

  /// get the flavor of this calibration detid
  CalibDetType calibFlavor() const { return (CalibDetType)((id_>>17)&0x7); }

  /// get the rbx number (if relevant)
  int rbx() const;
  /// get the sector identifier (if relevant)
  SectorId sector() const;
  /// get the sector identifier as a string (if relevant)
  std::string sectorString() const;
  /// get the calibration box channel (if relevant)
  int cboxChannel() const;
  /// get the calibration box channel as a string (if relevant)
  std::string cboxChannelString() const;

  /// constants
  static const int cbox_MixerHigh     = 1; // HB/HE/HO/HF
  static const int cbox_MixerLow      = 2; // HB/HE/HO/HF
  static const int cbox_MixerScint    = 3; // in HF only!
  static const int cbox_LaserMegatile = 4; // in HB/HE only!
  static const int cbox_RadDam1       = 5; // in HE only!
  static const int cbox_RadDam2       = 6; // in HE only!
  static const int cbox_RadDam3       = 7; // in HE only!

};

std::ostream& operator<<(std::ostream& s,const HcalCalibDetId& id);

#endif
