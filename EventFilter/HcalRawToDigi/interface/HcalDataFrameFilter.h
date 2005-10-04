#ifndef HCALDATAFRAMEFILTER_H
#define HCALDATAFRAMEFILTER_H 1

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

/** \class HcalDataFrameFilter
    
    Utility algorithm for filtering out digis from testbeam, etc where
    no zero-suppression was applied.  The digis can be required to
    have correct form (capid rotation, error bit off, data-valid bit
    on).  It can also be filtered by simple amplitude requirements.
    As these are applied in units proportional to energy, rather than
    transverse energy, and no calibration is applied, care should be used.
   
   $Date: 2005/07/26 15:10:51 $
   $Revision: 1.1 $
   \author J. Mans - Minnesota
*/
class HcalDataFrameFilter {
public:
  HcalDataFrameFilter(bool requireCapid, bool requireDVER, bool energyFilter, int firstSample=-1, int lastSample=-1, double minAmpl=-1);
  /// filter HB/HE data frames
  HBHEDigiCollection filter(const HBHEDigiCollection& incol);
  /// filter HF data frames
  HFDigiCollection filter(const HFDigiCollection& incol);
  /// filter HO data frames
  HODigiCollection filter(const HODigiCollection& incol);
  /// whether any filters are on
  bool active() const;
private:
  bool requireCapid_;
  bool requireDVER_;
  bool energyFilter_;
  int firstSample_, lastSample_;
  double minimumAmplitude_;
};



#endif
