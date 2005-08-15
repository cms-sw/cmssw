//
// F.Ratnikov (UMd), Jul. 19, 2005
//
#ifndef HcalDbService_h
#define HcalDbService_h

/**

   \class HcalDbService
   \brief Interface to fetch data from DB
   \author Fedor Ratnikov
   
*/

#include "DataFormats/HcalDetId/interface/HcalDetId.h"

class HcalDbService {
 public:
  // identify itself
  virtual const char* name () const = 0;
  
  // basic conversion function for single range (0<=count<32)
  virtual double adcShape (int fCount) const = 0;
  // bin size for the QIE conversion
  virtual double adcShapeBin (int fCount) const = 0;
  // pedestal  
  virtual double pedestal (const cms::HcalDetId& fCell, int fCapId) const = 0;
  // gain
  virtual double gain (const cms::HcalDetId& fCell, int fCapId) const = 0;
  // pedestal width
  virtual double pedestalError (const cms::HcalDetId& fCell, int fCapId) const = 0;
  // gain width
  virtual double gainError (const cms::HcalDetId& fCell, int fCapId) const = 0;
  // offset for the (cell,capId,range)
  virtual double offset (const cms::HcalDetId& fCell, int fCapId, int fRange) const = 0;
  // slope for the (cell,capId,range)
  virtual double slope (const cms::HcalDetId& fCell, int fCapId, int fRange) const = 0;

  // clone itself
  virtual HcalDbService* clone () const = 0;
};
#endif
