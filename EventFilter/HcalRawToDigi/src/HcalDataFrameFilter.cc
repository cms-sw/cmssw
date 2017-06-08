#include "EventFilter/HcalRawToDigi/interface/HcalDataFrameFilter.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"

namespace HcalDataFrameFilter_impl {

  template <class DataFrame> 
  bool check(const DataFrame& df, bool capcheck, bool dvercheck) {
    if (capcheck || dvercheck) {
      int lastcapid=0, capid=0;
      for (int i=0; i<df.size(); i++) {
	capid=df[i].capid();
	if (capcheck && i!=0 && ((lastcapid+1)%4)!=capid) 
	  return false;
	if (dvercheck && ( df[i].er() || !df[i].dv() ))
	  return false;
	lastcapid=capid;      
      }
    }
    return true;
  }

  template<>
  bool check<QIE10DataFrame>(const QIE10DataFrame& df, bool capcheck, bool linkerrcheck) {
    if (linkerrcheck && df.linkError()) return false;
    if (capcheck) {
      for (int i=0; i<df.samples(); i++) {
	if (!df[i].ok()) return false;
      }
    }
    return true;
  }

  template<>
  bool check<QIE11DataFrame>(const QIE11DataFrame& df, bool capcheck, bool linkerrcheck) {
    if (linkerrcheck && df.linkError()) return false;
    if (capcheck && df.capidError()) return false;
    return true;
  }


  template <class DataFrame> 
  double energySum(const DataFrame& df, int fs, int ls, const HcalDbService* conditions=nullptr) {
    double es=0;
    for (int i=fs; i<=ls && i<=df.size(); i++) 
      es+=df[i].nominal_fC();
    return es;
  }

  template <>
  double energySum<QIE11DataFrame>(const QIE11DataFrame& df, int fs, int ls, const HcalDbService* conditions) {
    const HcalQIECoder* channelCoder = conditions->getHcalCoder(df.id());
    const HcalQIEShape* shape = conditions->getHcalShape(channelCoder);
    CaloSamples tool;
    HcalCoderDb coder(*channelCoder, *shape);
    coder.adc2fC(df, tool);
    double es=0;
    for (int i=fs; i<=ls && i<=(int)df.samples(); i++)
      es+=tool[i];
    return es;
  }

}


HcalDataFrameFilter::HcalDataFrameFilter(bool requireCapid, bool requireDVER, bool energyFilter, int firstSample, int lastSample, double minAmpl) :
  requireCapid_(requireCapid), requireDVER_(requireDVER), energyFilter_(energyFilter),
  firstSample_(firstSample), lastSample_(lastSample), minimumAmplitude_(minAmpl), conditions_(nullptr) {
}

void HcalDataFrameFilter::setConditions(const HcalDbService* conditions) {
  conditions_ = conditions;
}

HBHEDigiCollection HcalDataFrameFilter::filter(const HBHEDigiCollection& incol, HcalUnpackerReport& r) {
  HBHEDigiCollection output;
  for (HBHEDigiCollection::const_iterator i=incol.begin(); i!=incol.end(); i++) {
    if (!HcalDataFrameFilter_impl::check(*i,requireCapid_,requireDVER_)) 
      r.countBadQualityDigi(i->id());
    else if (!energyFilter_ || minimumAmplitude_<HcalDataFrameFilter_impl::energySum(*i,firstSample_,lastSample_))
      output.push_back(*i);
  }
  return output;
}


HODigiCollection HcalDataFrameFilter::filter(const HODigiCollection& incol, HcalUnpackerReport& r) {
  HODigiCollection output;
  for (HODigiCollection::const_iterator i=incol.begin(); i!=incol.end(); i++) {
    if (!HcalDataFrameFilter_impl::check(*i,requireCapid_,requireDVER_))
      r.countBadQualityDigi(i->id());
    else if (!energyFilter_ || minimumAmplitude_<HcalDataFrameFilter_impl::energySum(*i,firstSample_,lastSample_))
      output.push_back(*i);
    
  }
  return output;
}

HcalCalibDigiCollection HcalDataFrameFilter::filter(const HcalCalibDigiCollection& incol, HcalUnpackerReport& r) {
  HcalCalibDigiCollection output;
  for (HcalCalibDigiCollection::const_iterator i=incol.begin(); i!=incol.end(); i++) {
    if (!HcalDataFrameFilter_impl::check(*i,requireCapid_,requireDVER_))
      r.countBadQualityDigi(i->id());
    else if (!energyFilter_ || minimumAmplitude_<HcalDataFrameFilter_impl::energySum(*i,firstSample_,lastSample_))
      output.push_back(*i);
    
  }
  return output;
}

HFDigiCollection HcalDataFrameFilter::filter(const HFDigiCollection& incol, HcalUnpackerReport& r) {
  HFDigiCollection output;
  for (HFDigiCollection::const_iterator i=incol.begin(); i!=incol.end(); i++) {
    if (!HcalDataFrameFilter_impl::check(*i,requireCapid_,requireDVER_))
      r.countBadQualityDigi(i->id());
    else if (!energyFilter_ || minimumAmplitude_<HcalDataFrameFilter_impl::energySum(*i,firstSample_,lastSample_))
      output.push_back(*i);    
  }
  return output;
}

ZDCDigiCollection HcalDataFrameFilter::filter(const ZDCDigiCollection& incol, HcalUnpackerReport& r) {
  ZDCDigiCollection output;
  for (ZDCDigiCollection::const_iterator i=incol.begin(); i!=incol.end(); i++) {
    if (!HcalDataFrameFilter_impl::check(*i,requireCapid_,requireDVER_))
      r.countBadQualityDigi(i->id());
    else if (!energyFilter_ || minimumAmplitude_<HcalDataFrameFilter_impl::energySum(*i,firstSample_,lastSample_))
      output.push_back(*i);    
  }
  return output;
}

QIE10DigiCollection HcalDataFrameFilter::filter(const QIE10DigiCollection& incol, HcalUnpackerReport& r) {
  QIE10DigiCollection output(incol.samples());
  for (QIE10DigiCollection::const_iterator i=incol.begin(); i!=incol.end(); i++) {
    QIE10DataFrame df(*i);
    if (!HcalDataFrameFilter_impl::check(df,requireCapid_,requireDVER_))
      r.countBadQualityDigi(i->id());
    // Never exclude QIE10 digis as their absence would be
    // treated as a digi with zero charged deposited in that channel
    output.push_back(df);
  }
  return output;
}

QIE11DigiCollection HcalDataFrameFilter::filter(const QIE11DigiCollection& incol, HcalUnpackerReport& r) {
  QIE11DigiCollection output(incol.samples());
  for (QIE11DigiCollection::const_iterator i=incol.begin(); i!=incol.end(); i++) {
    QIE11DataFrame df(*i);
    if (!HcalDataFrameFilter_impl::check(df,requireCapid_,requireDVER_))
      r.countBadQualityDigi(i->id());
    else if (!energyFilter_ || minimumAmplitude_<HcalDataFrameFilter_impl::energySum(df,firstSample_,lastSample_,conditions_))
      output.push_back(df);
  }
  return output;
}


bool HcalDataFrameFilter::active() const {
  return requireCapid_|requireDVER_|energyFilter_;
}

