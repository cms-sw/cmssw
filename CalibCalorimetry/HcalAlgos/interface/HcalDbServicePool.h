//
// F.Ratnikov (UMd), Jul. 19, 2005
//
#ifndef HcalDbServicePool_h
#define HcalDbServicePool_h

#include "boost/shared_ptr.hpp"

#include "CalibFormats/HcalObjects/interface/HcalDbServiceBase.h"

class HcalPedestals;
class HcalPedestalWidths;
class HcalGains;
class HcalGainWidths;

/**

   \class HcalDbServicePool
   \brief Pool implementation of the interface to fetch data from DB
   \author Fedor Ratnikov
   
*/
class HcalDbServicePool : public HcalDbServiceBase {
 public:
  HcalDbServicePool ();
  virtual ~HcalDbServicePool ();
  
  
  // identify itself
  virtual const char* name () const;
  
  // basic conversion function for single range (0<=count<32)
  virtual double adcShape (int fCount) const;
  // bin size for the QIE conversion
  virtual double adcShapeBin (int fCount) const;
  // pedestal  
  virtual const float* pedestals (const cms::HcalDetId& fCell) const;
  // gain
  virtual const float* gains (const cms::HcalDetId& fCell) const;
  // pedestal width
  virtual const float* pedestalErrors (const cms::HcalDetId& fCell) const;
  // gain width
  virtual const float* gainErrors (const cms::HcalDetId& fCell) const;
  // offset for the (cell,capId,range)
  virtual const float* offsets (const cms::HcalDetId& fCell) const;
  // slope for the (cell,capId,range)
  virtual const float* slopes (const cms::HcalDetId& fCell) const;

  // setters
  void setPedestals (const HcalPedestals* fPedestals) {mPedestals = fPedestals;}
  void setPedestalWidths (const HcalPedestalWidths* fPedestalWidths) {mPedestalWidths = fPedestalWidths;}
  void setGains (const HcalGains* fGains) {mGains = fGains;}
  void setGainWidths (const HcalGainWidths* fGainWidths) {mGainWidths = fGainWidths;}
  
  // clone itself
  virtual HcalDbServiceBase* clone () const;
  
 protected:
  
  const HcalPedestals* mPedestals;
  const HcalPedestalWidths* mPedestalWidths;
  const HcalGains* mGains;
  const HcalGainWidths* mGainWidths;
  
  const HcalDbServiceBase* mDefault;
};
#endif
