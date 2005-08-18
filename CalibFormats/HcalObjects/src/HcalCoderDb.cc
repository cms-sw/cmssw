/** \class HcalNominalCoder
    
    coder which uses DB services to convert to fC
    $Author: ratnikov
    $Date: 2005/08/04 18:16:29 $
    $Revision: 1.1 $
*/

#include "CalibFormats/HcalObjects/interface/HcalChannelCoder.h"
#include "CalibFormats/HcalObjects/interface/QieShape.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"

HcalCoderDb::HcalCoderDb (const HcalChannelCoder& fCoder, const QieShape& fShape)
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
    df.setSample(i, cms::HcalQIESample(mCoder->adc(*mShape, clf[i], capId), capId, 0, 0));
  }
}


void HcalCoderDb::adc2fC(const cms::HBHEDataFrame& df, CaloSamples& lf) const {adc2fC_ (df, lf);}
void HcalCoderDb::adc2fC(const cms::HODataFrame& df, CaloSamples& lf) const {adc2fC_ (df, lf);}
void HcalCoderDb::adc2fC(const cms::HFDataFrame& df, CaloSamples& lf) const {adc2fC_ (df, lf);}

void HcalCoderDb::fC2adc(const CaloSamples& clf, cms::HBHEDataFrame& df) const {fC2adc_ (clf, df);}
void HcalCoderDb::fC2adc(const CaloSamples& clf, cms::HFDataFrame& df) const {fC2adc_ (clf, df);}
void HcalCoderDb::fC2adc(const CaloSamples& clf, cms::HODataFrame& df) const {fC2adc_ (clf, df);}

