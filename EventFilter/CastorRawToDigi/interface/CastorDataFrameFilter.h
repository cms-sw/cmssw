#ifndef CASTORDATAFRAMEFILTER_H
#define CASTORDATAFRAMEFILTER_H 1

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"

/** \class CastorDataFrameFilter
    
    Utility algorithm for filtering out digis from testbeam, etc where
    no zero-suppression was applied.  The digis can be required to
    have correct form (capid rotation, error bit off, data-valid bit
    on).  It can also be filtered by simple amplitude requirements.
    As these are applied in units proportional to energy, rather than
    transverse energy, and no calibration is applied, care should be used.
   
   $Date: 2008/06/19 09:03:17 $
   $Revision: 1.1 $
   \author J. Mans - Minnesota
*/
class CastorDataFrameFilter {
public:
  CastorDataFrameFilter(bool requireCapid, bool requireDVER, bool energyFilter, int firstSample=-1, int lastSample=-1, double minAmpl=-1);
  /// filter Castor data frames
  CastorDigiCollection filter(const CastorDigiCollection& incol, HcalUnpackerReport& r);
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
