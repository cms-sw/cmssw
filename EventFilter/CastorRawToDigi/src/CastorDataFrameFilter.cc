#include "EventFilter/CastorRawToDigi/interface/CastorDataFrameFilter.h"

namespace CastorDataFrameFilter_impl {

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


CastorDataFrameFilter::CastorDataFrameFilter(bool requireCapid, bool requireDVER, bool energyFilter, int firstSample, int lastSample, double minAmpl) :
  requireCapid_(requireCapid), requireDVER_(requireDVER), energyFilter_(energyFilter),
  firstSample_(firstSample), lastSample_(lastSample), minimumAmplitude_(minAmpl) {
}

CastorDigiCollection CastorDataFrameFilter::filter(const CastorDigiCollection& incol, HcalUnpackerReport& r) {
  CastorDigiCollection output;
  for (CastorDigiCollection::const_iterator i=incol.begin(); i!=incol.end(); i++) {
    if (!CastorDataFrameFilter_impl::check(*i,requireCapid_,requireDVER_)) 
      r.countBadQualityDigi();
    else if (!energyFilter_ || minimumAmplitude_<CastorDataFrameFilter_impl::energySum(*i,firstSample_,lastSample_))
      output.push_back(*i);
  }
  return output;
}

bool CastorDataFrameFilter::active() const {
  return requireCapid_|requireDVER_|energyFilter_;
}

