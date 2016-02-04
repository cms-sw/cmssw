#include "CalibFormats/CastorObjects/interface/CastorNominalCoder.h"

void CastorNominalCoder::adc2fC(const CastorDataFrame& df, CaloSamples& lf) const {
  lf=CaloSamples(df.id(),df.size());
  for (int i=0; i<df.size(); i++) lf[i]=df[i].nominal_fC();
  lf.setPresamples(df.presamples());
}


namespace CastorNominalCoderTemplate {
  template <class Digi>
  void process(const CaloSamples& clf, Digi& df, int fCapIdOffset) {
    df=Digi(clf.id());
    df.setSize(clf.size());
    df.setPresamples(clf.presamples());
    for (int i=0; i<clf.size(); i++) {
      int capId = (fCapIdOffset + i) % 4;
      for (int q=1; q<128; q++) {
	df.setSample(i,HcalQIESample(q,capId,0,0));
	if (df[i].nominal_fC()>clf[i]) {
	  df.setSample(i,HcalQIESample(q-1,capId,0,0));
	  break;
	}
      }
    }
  }
}

void CastorNominalCoder::fC2adc(const CaloSamples& clf, CastorDataFrame& df, int fCapIdOffset) const {
  CastorNominalCoderTemplate::process(clf,df, fCapIdOffset);
}

