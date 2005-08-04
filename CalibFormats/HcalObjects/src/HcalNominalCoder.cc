#include "CalibFormats/HcalObjects/interface/HcalNominalCoder.h"

void HcalNominalCoder::adc2fC(const cms::HBHEDataFrame& df, CaloSamples& lf) const {
  lf=CaloSamples(df.id(),df.size());
  for (int i=0; i<df.size(); i++) lf[i]=df[i].nominal_fC();
  lf.setPresamples(df.presamples());
}
void HcalNominalCoder::adc2fC(const cms::HODataFrame& df, CaloSamples& lf) const {
  lf=CaloSamples(df.id(),df.size());
  for (int i=0; i<df.size(); i++) lf[i]=df[i].nominal_fC();
  lf.setPresamples(df.presamples());
}
void HcalNominalCoder::adc2fC(const cms::HFDataFrame& df, CaloSamples& lf) const {
  lf=CaloSamples(df.id(),df.size());
  for (int i=0; i<df.size(); i++) lf[i]=df[i].nominal_fC();
  lf.setPresamples(df.presamples());
}

namespace HcalNominalCoderTemplate {
  template <class Digi>
  void process(const CaloSamples& clf, Digi& df) {
    df=Digi(clf.id());
    df.setSize(clf.size());
    df.setPresamples(clf.presamples());
    for (int i=0; i<clf.size(); i++) 
      for (int q=1; q<128; q++) {
	df.setSample(i,cms::HcalQIESample(q,i%4,0,0));
	if (df[i].nominal_fC()>clf[i]) {
	  df.setSample(i,cms::HcalQIESample(q-1,i%4,0,0));
	  break;
	}
      }
  }
}

void HcalNominalCoder::fC2adc(const CaloSamples& clf, cms::HBHEDataFrame& df) const {
  HcalNominalCoderTemplate::process(clf,df);
}
void HcalNominalCoder::fC2adc(const CaloSamples& clf, cms::HFDataFrame& df) const {
  HcalNominalCoderTemplate::process(clf,df);
}
void HcalNominalCoder::fC2adc(const CaloSamples& clf, cms::HODataFrame& df) const {
  HcalNominalCoderTemplate::process(clf,df);
}
