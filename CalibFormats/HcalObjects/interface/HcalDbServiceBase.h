//
// F.Ratnikov (UMd), Jul. 19, 2005
//
#ifndef HcalDbServiceBase_h
#define HcalDbServiceBase_h

/**

   \class HcalDbServiceBase
   \brief Interface to fetch data from DB
   \author Fedor Ratnikov
   
*/

#include "DataFormats/HcalDetId/interface/HcalDetId.h"

class HcalDbServiceBase {
 public:
  // identify itself
  virtual const char* name () const = 0;
  
  // basic conversion function for single range (0<=count<32)
  virtual double adcShape (int fCount) const = 0;
  // bin size for the QIE conversion
  virtual double adcShapeBin (int fCount) const = 0;
  // pedestal  
  virtual const float* pedestals (const HcalDetId& fCell) const = 0;
  // gain
  virtual const float* gains (const HcalDetId& fCell) const = 0;
  // pedestal width
  virtual const float* pedestalErrors (const HcalDetId& fCell) const = 0;
  // gain width
  virtual const float* gainErrors (const HcalDetId& fCell) const = 0;
  // offset for the (cell,capId,range)
  virtual const float* offsets (const HcalDetId& fCell) const = 0;
  // slope for the (cell,capId,range)
  virtual const float* slopes (const HcalDetId& fCell) const = 0;
  // coding capId x Range into float[16]
  static int index (int fCapId, int Range) {return fCapId*4+Range;}

  // clone itself
  virtual HcalDbServiceBase* clone () const = 0;
};
#endif
