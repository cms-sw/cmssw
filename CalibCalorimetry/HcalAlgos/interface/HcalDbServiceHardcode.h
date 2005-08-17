//
// F.Ratnikov (UMd), Jul. 19, 2005
//
#ifndef HcalDbServiceHardcode_h
#define HcalDbServiceHardcode_h

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

/**

   \class HcalDbServiceHardcode
   \brief Hardcode implementation of the interface to fetch data from DB
   \author Fedor Ratnikov
   
*/
class HcalDbServiceHardcode : public HcalDbService {
 public:
  HcalDbServiceHardcode ();
  virtual ~HcalDbServiceHardcode ();

  virtual const char* name () const;

  // basic conversion function for single range (0<=count<32)
  virtual double adcShape (int fCount) const;
  // bin size for the QIE conversion
  virtual double adcShapeBin (int fCount) const;
  // pedestal  
  virtual double pedestal (const cms::HcalDetId& fCell, int fCapId) const;
  // gain
  virtual double gain (const cms::HcalDetId& fCell, int fCapId) const;
  // pedestal width
  virtual double pedestalError (const cms::HcalDetId& fCell, int fCapId) const;
  // gain width
  virtual double gainError (const cms::HcalDetId& fCell, int fCapId) const;
  // offset for the (cell,capId,range)
  virtual double offset (const cms::HcalDetId& fCell, int fCapId, int fRange) const;
  // slope for the (cell,capId,range)
  virtual double slope (const cms::HcalDetId& fCell, int fCapId, int fRange) const;

  virtual HcalDbService* clone () const;
};
#endif
