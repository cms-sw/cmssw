#ifndef HCALNOMINALCODER_H
#define HCALNOMINALCODER_H 1

#include "CalibFormats/HcalObjects/interface/HcalCoder.h"

/** \class HcalNominalCoder
    
    Simple coder which uses the QIESample to convert to fC

   $Date: 2011/01/21 22:24:37 $
   $Revision: 1.4.6.1 $
*/
class HcalNominalCoder : public HcalCoder {
public:
  virtual void adc2fC(const HBHEDataFrame& df, CaloSamples& lf) const;
  virtual void adc2fC(const HODataFrame& df, CaloSamples& lf) const;
  virtual void adc2fC(const HFDataFrame& df, CaloSamples& lf) const;
  virtual void adc2fC(const ZDCDataFrame& df, CaloSamples& lf) const;
  virtual void adc2fC(const HcalCalibDataFrame& df, CaloSamples& lf) const;
  virtual void fC2adc(const CaloSamples& clf, HBHEDataFrame& df, int fCapIdOffset) const;
  virtual void fC2adc(const CaloSamples& clf, HFDataFrame& df, int fCapIdOffset) const;
  virtual void fC2adc(const CaloSamples& clf, HODataFrame& df, int fCapIdOffset) const;
  virtual void fC2adc(const CaloSamples& clf, ZDCDataFrame& df, int fCapIdOffset) const;
  virtual void fC2adc(const CaloSamples& clf, HcalCalibDataFrame& df, int fCapIdOffset) const;

private:

  virtual void adc2fC(const HcalUpgradeDataFrame& df, CaloSamples& lf) const { }
  virtual void fC2adc(const CaloSamples& clf, HcalUpgradeDataFrame& df, int fCapIdOffset) const { }

};

#endif
