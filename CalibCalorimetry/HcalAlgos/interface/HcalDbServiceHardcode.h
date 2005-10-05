//
// F.Ratnikov (UMd), Jul. 19, 2005
//
#ifndef HcalDbServiceHardcode_h
#define HcalDbServiceHardcode_h

#include "CalibFormats/HcalObjects/interface/HcalDbServiceBase.h"

/**

   \class HcalDbServiceHardcode
   \brief Hardcode implementation of the interface to fetch data from DB
   \author Fedor Ratnikov
   
*/
class HcalDbServiceHardcode : public HcalDbServiceBase {
 public:
  HcalDbServiceHardcode ();
  virtual ~HcalDbServiceHardcode ();

  // identify itself
  virtual const char* name () const;
  
  // basic conversion function for single range (0<=count<32)
  virtual double adcShape (int fCount) const;
  // bin size for the QIE conversion
  virtual double adcShapeBin (int fCount) const;
  // pedestal  
  virtual const float* pedestals (const HcalDetId& fCell) const;
  // gain
  virtual const float* gains (const HcalDetId& fCell) const;
  // pedestal width
  virtual const float* pedestalErrors (const HcalDetId& fCell) const;
  // gain const width
  virtual const float* gainErrors (const HcalDetId& fCell) const;
  // offset for the (cell,capId,range)
  virtual const float* offsets (const HcalDetId& fCell) const;
  // slope for the (cell,capId,range)
  virtual const float* slopes (const HcalDetId& fCell) const;

  // clone itself
  virtual HcalDbServiceBase* clone () const;
  
 private:
  // internal buffers
  mutable float pedestal [4];
  mutable float gain [4];
  mutable float pError [4];
  mutable float gError [4];
  mutable float offset [16];
  mutable float slope [16];
};
#endif
