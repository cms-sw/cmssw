#include "EventFilter/HcalRawToDigi/interface/HcalDataFrameFilter.h"

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

  template <class DataFrame> 
  double energySum(const DataFrame& df, int fs, int ls) {
    double es=0;
    for (int i=fs; i<=ls && i<=df.size(); i++) 
      es+=df[i].nominal_fC();
    return es;
  }

}


HcalDataFrameFilter::HcalDataFrameFilter(bool requireCapid, bool requireDVER, bool energyFilter, int firstSample, int lastSample, double minAmpl) :
  requireCapid_(requireCapid), requireDVER_(requireDVER), energyFilter_(energyFilter),
  firstSample_(firstSample), lastSample_(lastSample), minimumAmplitude_(minAmpl) {
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


bool HcalDataFrameFilter::active() const {
  return requireCapid_|requireDVER_|energyFilter_;
}

