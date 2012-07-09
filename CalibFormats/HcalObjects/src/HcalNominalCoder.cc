#include "CalibFormats/HcalObjects/interface/HcalNominalCoder.h"

void HcalNominalCoder::adc2fC(const HBHEDataFrame& df, CaloSamples& lf) const {
  lf=CaloSamples(df.id(),df.size());
  for (int i=0; i<df.size(); i++) lf[i]=df[i].nominal_fC();
  lf.setPresamples(df.presamples());
}
void HcalNominalCoder::adc2fC(const HODataFrame& df, CaloSamples& lf) const {
  lf=CaloSamples(df.id(),df.size());
  for (int i=0; i<df.size(); i++) lf[i]=df[i].nominal_fC();
  lf.setPresamples(df.presamples());
}
void HcalNominalCoder::adc2fC(const HFDataFrame& df, CaloSamples& lf) const {
  lf=CaloSamples(df.id(),df.size());
  for (int i=0; i<df.size(); i++) lf[i]=df[i].nominal_fC();
  lf.setPresamples(df.presamples());
}
void HcalNominalCoder::adc2fC(const ZDCDataFrame& df, CaloSamples& lf) const {
  lf=CaloSamples(df.id(),df.size());
  for (int i=0; i<df.size(); i++) lf[i]=df[i].nominal_fC();
  lf.setPresamples(df.presamples());
}
void HcalNominalCoder::adc2fC(const HcalCalibDataFrame& df, CaloSamples& lf) const {
  lf=CaloSamples(df.id(),df.size());
  for (int i=0; i<df.size(); i++) lf[i]=df[i].nominal_fC();
  lf.setPresamples(df.presamples());
}

namespace HcalNominalCoderTemplate {
  template <class Digi>
  void process(const CaloSamples& clf, Digi& df, int fCapIdOffset) {
    df=Digi(clf.id());
    df.setSize(clf.size());
    df.setPresamples(clf.presamples());
    for (int i=0; i<clf.size(); i++) {
      int capId = (fCapIdOffset + i) % 4;
      for (int q=1; q<256; q++) {
	df.setSample(i,HcalQIESample(q,capId,0,0));
	if (df[i].nominal_fC()>clf[i]) {
	  df.setSample(i,HcalQIESample(q-1,capId,0,0));
	  break;
	}
      }
    }
  }
}

void HcalNominalCoder::fC2adc(const CaloSamples& clf, HBHEDataFrame& df, int fCapIdOffset) const {
  HcalNominalCoderTemplate::process(clf,df, fCapIdOffset);
}
void HcalNominalCoder::fC2adc(const CaloSamples& clf, HFDataFrame& df, int fCapIdOffset) const {
  HcalNominalCoderTemplate::process(clf,df, fCapIdOffset);
}
void HcalNominalCoder::fC2adc(const CaloSamples& clf, HODataFrame& df, int fCapIdOffset) const {
  HcalNominalCoderTemplate::process(clf,df, fCapIdOffset);
}
void HcalNominalCoder::fC2adc(const CaloSamples& clf, ZDCDataFrame& df, int fCapIdOffset) const {
  HcalNominalCoderTemplate::process(clf,df, fCapIdOffset);
}
void HcalNominalCoder::fC2adc(const CaloSamples& clf, HcalCalibDataFrame& df, int fCapIdOffset) const {
  HcalNominalCoderTemplate::process(clf,df, fCapIdOffset);
}
