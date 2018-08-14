#ifndef HCALDATAFRAMEFILTER_H
#define HCALDATAFRAMEFILTER_H 1

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

/** \class HcalDataFrameFilter
    
    Utility algorithm for filtering out digis from testbeam, etc where
    no zero-suppression was applied.  The digis can be required to
    have correct form (capid rotation, error bit off, data-valid bit
    on).  It can also be filtered by simple amplitude requirements.
    As these are applied in units proportional to energy, rather than
    transverse energy, and no calibration is applied, care should be used.
   
   \author J. Mans - Minnesota
*/
class HcalDataFrameFilter {
public:
  HcalDataFrameFilter(bool requireCapid, bool requireDVER, bool energyFilter, int firstSample=-1, int lastSample=-1, double minAmpl=-1);
  /// filter HB/HE data frames
  HBHEDigiCollection filter(const HBHEDigiCollection& incol, HcalUnpackerReport& r);
  /// filter HF data frames
  HFDigiCollection filter(const HFDigiCollection& incol, HcalUnpackerReport& r);
  /// filter HO data frames
  HODigiCollection filter(const HODigiCollection& incol, HcalUnpackerReport& r);
  /// filter HO data frames
  HcalCalibDigiCollection filter(const HcalCalibDigiCollection& incol, HcalUnpackerReport& r);
  /// filter ZDC data frames
  ZDCDigiCollection filter(const ZDCDigiCollection& incol, HcalUnpackerReport& r);
  /// filter QIE10 data frames
  QIE10DigiCollection filter(const QIE10DigiCollection& incol, HcalUnpackerReport& r);
  /// filter QIE11 data frames
  QIE11DigiCollection filter(const QIE11DigiCollection& incol, HcalUnpackerReport& r);
  /// whether any filters are on
  bool active() const;
  /// get conditions
  void setConditions(const HcalDbService* conditions);
private:
  bool requireCapid_;
  bool requireDVER_;
  bool energyFilter_;
  int firstSample_, lastSample_;
  double minimumAmplitude_;
  const HcalDbService* conditions_;
};



#endif
