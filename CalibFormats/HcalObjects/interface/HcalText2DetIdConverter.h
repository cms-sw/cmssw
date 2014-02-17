#ifndef HCALTEXT2DETIDCONVERTER_H
#define HCALTEXT2DETIDCONVERTER_H

/** \class HcalText2DetIdConverter
    \author F.Ratnikov, UMd
   Converts any flavour of HcalDetId to/from ascii strings


   For Calibration DetIds:

   Flavor = "CALIB_[HB|HE|HO|HF]" or "HOX"

   fField1 is ieta (-2:2 for CALIB)
   fField2 is iphi
   fField3 is ctype (CALIB only)

   $Id: HcalText2DetIdConverter.h,v 1.2 2008/07/22 11:41:34 mansj Exp $
*/
#include <string>
#include "DataFormats/DetId/interface/DetId.h"

class HcalText2DetIdConverter {
 public:

  HcalText2DetIdConverter (const std::string& fFlavor = "NA", const std::string& fField1 = "0",
			  const std::string& fField2 = "0", const std::string& fField3 = "0");
  
  HcalText2DetIdConverter (DetId fId);

  bool isHcalDetId () const;
  bool isHcalCalibDetId () const;
  bool isHcalTrigTowerDetId () const;
  bool isHcalZDCDetId () const;

  const std::string& getFlavor () const {return flavorName;}
  const std::string& getField1 () const {return field1;}
  const std::string& getField2 () const {return field2;}
  const std::string& getField3 () const {return field3;}
  DetId getId () const {return mId;}
  int getField (int i) const;
  std::string toString () const;
  
 private:
  bool init (const std::string& fFlavor, const std::string& fField1,
	     const std::string& fField2, const std::string& fField3);
  bool init (DetId fId);
  void setField (int i, int fValue);
  std::string flavorName;
  std::string field1;
  std::string field2;
  std::string field3;
  DetId mId;
};

#endif
