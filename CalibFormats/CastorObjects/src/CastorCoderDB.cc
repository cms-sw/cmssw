/** \class CastorCoderDB
    
    coder which uses DB services to convert to fC
*/

#include "CondFormats/CastorObjects/interface/CastorQIEShape.h"
#include "CondFormats/CastorObjects/interface/CastorQIECoder.h"

#include "CalibFormats/CastorObjects/interface/CastorCoderDb.h"

CastorCoderDb::CastorCoderDb (const CastorQIECoder& fCoder, const CastorQIEShape& fShape)
  : mCoder (&fCoder),
    mShape (&fShape)
{}

template <class Digi> void CastorCoderDb::adc2fC_ (const Digi& df, CaloSamples& clf) const {
  clf=CaloSamples(df.id(),df.size());
  for (int i=0; i<df.size(); i++) {
    clf[i]=mCoder->charge (*mShape, df[i].adc (), df[i].capid ());
  }
  clf.setPresamples(df.presamples());
}

template <class Digi> void CastorCoderDb::fC2adc_ (const CaloSamples& clf, Digi& df, int fCapIdOffset) const {
  df = Digi (clf.id ());
  df.setSize (clf.size ());
  df.setPresamples (clf.presamples ());
  for (int i=0; i<clf.size(); i++) {
    int capId = (fCapIdOffset + i) % 4;
    df.setSample(i, HcalQIESample(mCoder->adc(*mShape, clf[i], capId), capId, 0, 0));
  }
}

void CastorCoderDb::adc2fC(const CastorDataFrame& df, CaloSamples& lf) const {adc2fC_ (df, lf);}

void CastorCoderDb::fC2adc(const CaloSamples& clf, CastorDataFrame& df, int fCapIdOffset) const {fC2adc_ (clf, df, fCapIdOffset);}

