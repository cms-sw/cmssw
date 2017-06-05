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

  bool checkQIE10(const QIE10DataFrame& df, bool capcheck, bool linkerrcheck) {
    if (linkerrcheck && df.linkError()) return false;
    if (capcheck) {
      for (int i=0; i<df.samples(); i++) {
	if (!df[i].ok()) return false;
      }
    }
    return true;
  }

  bool checkQIE11(const QIE11DataFrame& df, bool capcheck, bool linkerrcheck) {
    if (linkerrcheck && df.linkError()) return false;
    if (capcheck && df.capidError()) return false;
    return true;
  }


  template <class DataFrame> 
  double energySum(const DataFrame& df, int fs, int ls) {
    double es=0;
    for (int i=fs; i<=ls && i<=df.size(); i++) 
      es+=df[i].nominal_fC();
    return es;
  }

  double const adc2fC_QIE11[256] = {
    1.58, 4.73, 7.88, 11.0, 14.2, 17.3, 20.5, 23.6,
    26.8, 29.9, 33.1, 36.2, 39.4, 42.5, 45.7, 48.8,
    53.6, 60.1, 66.6, 73.0, 79.5, 86.0, 92.5, 98.9,
    105, 112, 118, 125, 131, 138, 144, 151,
    157, 164, 170, 177, 186, 199, 212, 225,
    238, 251, 264, 277, 289, 302, 315, 328,
    341, 354, 367, 380, 393, 406, 418, 431,
    444, 464, 490, 516, 542, 568, 594, 620,
    569, 594, 619, 645, 670, 695, 720, 745,
    771, 796, 821, 846, 871, 897, 922, 947,
    960, 1010, 1060, 1120, 1170, 1220, 1270, 1320,
    1370, 1430, 1480, 1530, 1580, 1630, 1690, 1740,
    1790, 1840, 1890, 1940,  2020, 2120, 2230, 2330,
    2430, 2540, 2640, 2740, 2850, 2950, 3050, 3150,
    3260, 3360, 3460, 3570, 3670, 3770, 3880, 3980,
    4080, 4240, 4450, 4650, 4860, 5070, 5280, 5490,

    5080, 5280, 5480, 5680, 5880, 6080, 6280, 6480,
    6680, 6890, 7090, 7290, 7490, 7690, 7890, 8090,
    8400, 8810, 9220, 9630, 10000, 10400, 10900, 11300,
    11700, 12100, 12500, 12900, 13300, 13700, 14100, 14500,
    15000, 15400, 15800, 16200, 16800, 17600, 18400, 19300,
    20100, 20900, 21700, 22500, 23400, 24200, 25000, 25800,
    26600, 27500, 28300, 29100, 29900, 30700, 31600, 32400,
    33200, 34400, 36100, 37700, 39400, 41000, 42700, 44300,
    41100, 42700, 44300, 45900, 47600, 49200, 50800, 52500,
    54100, 55700, 57400, 59000, 60600, 62200, 63900, 65500,
    68000, 71300, 74700, 78000, 81400, 84700, 88000, 91400,
    94700, 98100, 101000, 105000, 108000, 111000, 115000, 118000,
    121000, 125000, 128000, 131000, 137000, 145000, 152000, 160000,
    168000, 176000, 183000, 191000, 199000, 206000, 214000, 222000,
    230000, 237000, 245000, 253000, 261000, 268000, 276000, 284000,
    291000, 302000, 316000, 329000, 343000, 356000, 370000, 384000
  };

  double energySumQIE11(const QIE11DataFrame& df, unsigned int fs, unsigned int ls) {
    double es=0;
    for (unsigned int i=fs; i<=ls && i<=df.size(); i++)
      es+=adc2fC_QIE11[df[i].adc()];
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

QIE10DigiCollection HcalDataFrameFilter::filter(const QIE10DigiCollection& incol, HcalUnpackerReport& r) {
  QIE10DigiCollection output;
  for (QIE10DigiCollection::const_iterator i=incol.begin(); i!=incol.end(); i++) {
    if (!HcalDataFrameFilter_impl::checkQIE10(*i,requireCapid_,requireDVER_))
      r.countBadQualityDigi(i->id());
    // Never exclude QIE10 digis as their absence would be
    // treated as a digi with zero charged deposited in that channel
    output.push_back(*i);
  }
  return output;
}

QIE11DigiCollection HcalDataFrameFilter::filter(const QIE11DigiCollection& incol, HcalUnpackerReport& r) {
  QIE11DigiCollection output;
  for (QIE11DigiCollection::const_iterator i=incol.begin(); i!=incol.end(); i++) {
    if (!HcalDataFrameFilter_impl::checkQIE11(*i,requireCapid_,requireDVER_))
      r.countBadQualityDigi(i->id());
    else if (!energyFilter_ || minimumAmplitude_<HcalDataFrameFilter_impl::energySumQIE11(*i,firstSample_,lastSample_))
      output.push_back(*i);
  }
  return output;
}


bool HcalDataFrameFilter::active() const {
  return requireCapid_|requireDVER_|energyFilter_;
}

