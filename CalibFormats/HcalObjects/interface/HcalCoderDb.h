#ifndef HCAL_CODER_DB_H
#define HCAL_CODER_DB_H

#include "CalibFormats/HcalObjects/interface/HcalCoder.h"

/** \class HcalCoderDb
    
    coder which uses DB services to convert to fC
    $Author: ratnikov
    $Date: 2013/04/19 07:36:50 $
    $Revision: 1.7 $
*/

class HcalQIECoder;
class HcalQIEShape;

class HcalCoderDb : public HcalCoder {
public:
  HcalCoderDb (const HcalQIECoder& fCoder, const HcalQIEShape& fShape);

  virtual void adc2fC(const HBHEDataFrame& df, CaloSamples& lf) const;
  virtual void adc2fC(const HODataFrame& df, CaloSamples& lf) const;
  virtual void adc2fC(const HFDataFrame& df, CaloSamples& lf) const;
  virtual void adc2fC(const ZDCDataFrame& df, CaloSamples& lf) const;
  virtual void adc2fC(const HcalCalibDataFrame& df, CaloSamples& lf) const;
  virtual void adc2fC(const HcalUpgradeDataFrame& df, CaloSamples& lf) const;

  virtual void fC2adc(const CaloSamples& clf, HBHEDataFrame& df, int fCapIdOffset) const;
  virtual void fC2adc(const CaloSamples& clf, HFDataFrame& df, int fCapIdOffset) const;
  virtual void fC2adc(const CaloSamples& clf, HODataFrame& df, int fCapIdOffset) const;
  virtual void fC2adc(const CaloSamples& clf, ZDCDataFrame& df, int fCapIdOffset) const;
  virtual void fC2adc(const CaloSamples& clf, HcalCalibDataFrame& df, int fCapIdOffset) const;
  virtual void fC2adc(const CaloSamples& clf, HcalUpgradeDataFrame& df, int fCapIdOffset) const;

private:
  template <class Digi> void adc2fC_ (const Digi& df, CaloSamples& clf) const;
  template <class Digi> void fC2adc_ (const CaloSamples& clf, Digi& df, int fCapIdOffset) const;
  template <class Digi> void fCUpgrade2adc_ (const CaloSamples& clf, Digi& df, int fCapIdOffset) const;

  const HcalQIECoder* mCoder;
  const HcalQIEShape* mShape;
};

#endif
