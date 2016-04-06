/** \class HcalCoderDB
    
    coder which uses DB services to convert to fC
    $Author: ratnikov
*/

#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "DataFormats/HcalDigi/interface/HcalUpgradeQIESample.h"

HcalCoderDb::HcalCoderDb (const HcalQIECoder& fCoder, const HcalQIEShape& fShape)
  : mCoder (&fCoder),
    mShape (&fShape)
{}

template <class Digi> void HcalCoderDb::adc2fC_ (const Digi& df, CaloSamples& clf) const {
  clf=CaloSamples(df.id(),df.size());
  for (int i=0; i<df.size(); i++) {
    clf[i]=mCoder->charge (*mShape, df[i].adc (), df[i].capid ());
  }
  clf.setPresamples(df.presamples());
}

template <> void HcalCoderDb::adc2fC_<QIE10DataFrame> (const QIE10DataFrame& df, CaloSamples& clf) const {
  clf=CaloSamples(df.id(),df.samples());
  for (int i=0; i<df.samples(); i++) {
    clf[i]=mCoder->charge (*mShape, df[i].adc (), df[i].capid ());
    if(df[i].soi()) clf.setPresamples(i);
  }
}

template <class Digi> void HcalCoderDb::fC2adc_ (const CaloSamples& clf, Digi& df, int fCapIdOffset) const {
  df = Digi (clf.id ());
  df.setSize (clf.size ());
  df.setPresamples (clf.presamples ());
  for (int i=0; i<clf.size(); i++) {
    int capId = (fCapIdOffset + i) % 4;
    df.setSample(i, HcalQIESample(mCoder->adc(*mShape, clf[i], capId), capId, 0, 0));
  }
}

template <class Digi> void HcalCoderDb::fCUpgrade2adc_ (const CaloSamples& clf, Digi& df, int fCapIdOffset) const {
  df = HcalUpgradeDataFrame(clf.id(), fCapIdOffset, clf.size(), 
			    clf.presamples());
  for (int i=0; i<clf.size(); ++i) {
    df.setSample(i, mCoder->adc(*mShape, clf[i], df.capId(i)),
		 0, true);
  }
}

template <> void HcalCoderDb::fC2adc_<QIE10DataFrame> (const CaloSamples& clf, QIE10DataFrame& df, int fCapIdOffset) const {
  int presample = clf.presamples ();
  for (int i=0; i<clf.size(); i++) {
    int capId = (fCapIdOffset + i) % 4;
	bool soi = (i==presample);
    df.setSample(i, mCoder->adc(*mShape, clf[i], capId), 0, 0, capId, soi, true);
  }
}

void HcalCoderDb::adc2fC(const HBHEDataFrame& df, CaloSamples& lf) const {adc2fC_ (df, lf);}
void HcalCoderDb::adc2fC(const HODataFrame& df, CaloSamples& lf) const {adc2fC_ (df, lf);}
void HcalCoderDb::adc2fC(const HFDataFrame& df, CaloSamples& lf) const {adc2fC_ (df, lf);}
void HcalCoderDb::adc2fC(const ZDCDataFrame& df, CaloSamples& lf) const {adc2fC_ (df, lf);}
void HcalCoderDb::adc2fC(const HcalCalibDataFrame& df, CaloSamples& lf) const {adc2fC_ (df, lf);}
void HcalCoderDb::adc2fC(const HcalUpgradeDataFrame& df, CaloSamples& lf) const {adc2fC_ (df, lf);}
void HcalCoderDb::adc2fC(const QIE10DataFrame& df, CaloSamples& lf) const {adc2fC_ (df, lf);}

void HcalCoderDb::fC2adc(const CaloSamples& clf, HBHEDataFrame& df, int fCapIdOffset) const {fC2adc_ (clf, df, fCapIdOffset);}
void HcalCoderDb::fC2adc(const CaloSamples& clf, HFDataFrame& df, int fCapIdOffset) const {fC2adc_ (clf, df, fCapIdOffset);}
void HcalCoderDb::fC2adc(const CaloSamples& clf, HODataFrame& df, int fCapIdOffset) const {fC2adc_ (clf, df, fCapIdOffset);}
void HcalCoderDb::fC2adc(const CaloSamples& clf, ZDCDataFrame& df, int fCapIdOffset) const {fC2adc_ (clf, df, fCapIdOffset);}
void HcalCoderDb::fC2adc(const CaloSamples& clf, HcalCalibDataFrame& df, int fCapIdOffset) const {fC2adc_ (clf, df, fCapIdOffset);}
void HcalCoderDb::fC2adc(const CaloSamples& clf, HcalUpgradeDataFrame& df, int fCapIdOffset) const {fCUpgrade2adc_ (clf, df, fCapIdOffset);}
void HcalCoderDb::fC2adc(const CaloSamples& clf, QIE10DataFrame& df, int fCapIdOffset) const {fC2adc_ (clf, df, fCapIdOffset);}

