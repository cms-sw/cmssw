#ifndef DATAFORMATS_HCALDETID_HCALCALIBDETID_H
#define DATAFORMATS_HCALDETID_HCALCALIBDETID_H 1

#include <ostream>
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalOtherDetId.h"

/** \class HcalCalibDetId
  *  
  *  Contents of the HcalCalibDetId :
  *     [19:17] Calibration Category (1 = CalibUnit, 2 = HX, 3=uMNio/qie, 4=CastorRad)
  *
  *  For CalibUnit:
  *     [16:14] Subdetector
  *     [13:11] Ieta (-2 -> 2)
  *     [10:4] Iphi ("low edge")
  *     [3:0] Calibration channel type
  * 
  *  For HX (HOCrosstalk) channels:
  *     [11] side (true = positive)
  *     [10:7] ieta
  *     [6:0] iphi
  *
  *  For uMNqie channels:
  *     [7:0] channel (typically just 0 or 1, but space for more if needed)
  *
  *  For Castor Radiation Facility:
  *     [16:10] RM
  *     [9:5] fiber-in-rm
  *     [4:0] channel-on-fiber
  *     
  *
  * \author J. Mans - Minnesota
  */
class HcalCalibDetId : public HcalOtherDetId {
public:
  /** Type identifier within calibration det ids */
  enum CalibDetType { CalibrationBox = 1, HOCrosstalk = 2, uMNqie = 3, CastorRadFacility = 4 };

  /** Create a null det id */
  HcalCalibDetId();
  /** Create from a raw id */
  HcalCalibDetId(uint32_t rawid);
  /** Create from a generic cell id */
  HcalCalibDetId(const DetId& id);
  HcalCalibDetId& operator=(const DetId& id);
  /** Construct a calibration box - channel detid */
  HcalCalibDetId(HcalSubdetector subdet, int ieta, int iphi, int ctype);
  /** Construct an HO Crosstalk id  */
  HcalCalibDetId(int ieta, int iphi);
  /** Construct a uMNqie id or other id which uses a single value plus a DetType */
  HcalCalibDetId(CalibDetType dt, int value);
    /** Construct a Castor radiation test facility id or other id which uses three values plus a DetType */
  HcalCalibDetId(CalibDetType dt, int value1, int value2, int value3);

  /// get the flavor of this calibration detid
  CalibDetType calibFlavor() const { return (CalibDetType)((id_>>17)&0x7); }


  /// get the HcalSubdetector (if relevant)
  HcalSubdetector hcalSubdet() const;
  /// get the rbx name (if relevant)
//  std::string rbx() const;
  /// get the "ieta" identifier (if relevant)
  int ieta() const;
  /// get the low-edge iphi (if relevant)
  int iphi() const;
  /// get the calibration box channel (if relevant)
  int cboxChannel() const;
  /// get the calibration box channel as a string (if relevant)
  std::string cboxChannelString() const;

  /// get the rm (where relevant)
  int rm() const;
  /// get the fiber (where relevant)
  int fiber() const;
  /// get the channel (for uMNio/qie or similar)
  int channel() const;

  /// get the sign of ieta (+/-1)
  int zside() const;

  /// constants
  static const int cbox_MixerHigh     = 0; // HB/HE/HO/HF
  static const int cbox_MixerLow      = 1; // HB/HE/HO/HF
  static const int cbox_LaserMegatile = 2; // in HB only!
  static const int cbox_RadDam_Layer0_RM4 = 3; // in HE only!
  static const int cbox_RadDam_Layer7_RM4 = 4; // in HE only!
  static const int cbox_RadDam_Layer0_RM1 = 5; // in HE only!
  static const int cbox_RadDam_Layer7_RM1 = 6; // in HE only!
  static const int cbox_HOCrosstalkPIN = 7; // in (part of) HO only!
  static const int cbox_HF_ScintillatorPIN = 8; // in HF only!
};

std::ostream& operator<<(std::ostream& s,const HcalCalibDetId& id);

#endif
