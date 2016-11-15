#ifndef HCALCODER_H
#define HCALCODER_H 1

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalCalibDataFrame.h"
#include "DataFormats/HcalDigi/interface/ZDCDataFrame.h"
#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"
#include "DataFormats/HcalDigi/interface/QIE11DataFrame.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"

/** \class HcalCoder
    
    Abstract interface of a coder/decoder which converts ADC values to
    and from femtocolumbs of collected charge.

*/
class HcalCoder {
public:
  virtual void adc2fC(const HBHEDataFrame& df, CaloSamples& lf) const = 0;
  virtual void adc2fC(const HODataFrame& df, CaloSamples& lf) const = 0;
  virtual void adc2fC(const HFDataFrame& df, CaloSamples& lf) const = 0;
  virtual void adc2fC(const ZDCDataFrame& df, CaloSamples& lf) const = 0;
  virtual void adc2fC(const HcalCalibDataFrame& df, CaloSamples& lf) const = 0;
  virtual void adc2fC(const QIE10DataFrame& df, CaloSamples& lf) const = 0;
  virtual void adc2fC(const QIE11DataFrame& df, CaloSamples& lf) const = 0;
  virtual void fC2adc(const CaloSamples& clf, HBHEDataFrame& df, int fCapIdOffset) const = 0;
  virtual void fC2adc(const CaloSamples& clf, HFDataFrame& df, int fCapIdOffset) const = 0;
  virtual void fC2adc(const CaloSamples& clf, HODataFrame& df, int fCapIdOffset) const = 0;
  virtual void fC2adc(const CaloSamples& clf, ZDCDataFrame& df, int fCapIdOffset) const = 0;
  virtual void fC2adc(const CaloSamples& clf, HcalCalibDataFrame& df, int fCapIdOffset) const = 0;
  virtual void fC2adc(const CaloSamples& clf, QIE10DataFrame& df, int fCapIdOffset) const = 0;
  virtual void fC2adc(const CaloSamples& clf, QIE11DataFrame& df, int fCapIdOffset) const = 0;
};

#endif
