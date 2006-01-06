/** \class HcalCoderDB
    
    coder which uses DB services to convert to fC
    $Author: ratnikov
    $Date: 2005/12/15 23:38:00 $
    $Revision: 1.3 $
*/

#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"

#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"

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

template <class Digi> void HcalCoderDb::fC2adc_ (const CaloSamples& clf, Digi& df, int fCapIdOffset) const {
  df = Digi (clf.id ());
  df.setSize (clf.size ());
  df.setPresamples (clf.presamples ());
  for (int i=0; i<clf.size(); i++) {
    int capId = (fCapIdOffset + i) % 4;
    df.setSample(i, HcalQIESample(mCoder->adc(*mShape, clf[i], capId), capId, 0, 0));
  }
}


void HcalCoderDb::adc2fC(const HBHEDataFrame& df, CaloSamples& lf) const {adc2fC_ (df, lf);}
void HcalCoderDb::adc2fC(const HODataFrame& df, CaloSamples& lf) const {adc2fC_ (df, lf);}
void HcalCoderDb::adc2fC(const HFDataFrame& df, CaloSamples& lf) const {adc2fC_ (df, lf);}

void HcalCoderDb::fC2adc(const CaloSamples& clf, HBHEDataFrame& df, int fCapIdOffset) const {fC2adc_ (clf, df, fCapIdOffset);}
void HcalCoderDb::fC2adc(const CaloSamples& clf, HFDataFrame& df, int fCapIdOffset) const {fC2adc_ (clf, df, fCapIdOffset);}
void HcalCoderDb::fC2adc(const CaloSamples& clf, HODataFrame& df, int fCapIdOffset) const {fC2adc_ (clf, df, fCapIdOffset);}

