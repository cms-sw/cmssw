#ifndef CALIBFORMATS_CALOTPG_HCALTPGSCALE_H
#define CALIBFORMATS_CALOTPG_HCALTPGSCALE_H 1

#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveSample.h"

namespace edm {
  class EventSetup;
}

/** \class HcalTPGScale
 *
 *  To be stored in the CaloTPGRecord
 *  
 * $Date: 2007/02/19 16:24:20 $
 * $Revision: 1.1 $
 * \author J. Mans - Minnesota
 */
class HcalTPGScale {
public:
  /** \brief nominal ET value of this sample/code as to be used in
      RCT LUT creation */
  virtual double et_RCT(const HcalTrigTowerDetId& id, const
			HcalTriggerPrimitiveSample& s) const = 0;
  /// smallest ET value which would resolve into this sample
  virtual double et_bin_low(const HcalTrigTowerDetId& id, const
			    HcalTriggerPrimitiveSample& s) const = 0;
  /** smallest ET value which would not resolve into sample (too big) */
  virtual double et_bin_high(const HcalTrigTowerDetId& id, const
		     HcalTriggerPrimitiveSample& s) const = 0;
  /// Get any needed information from the event setup
  virtual void setup(const edm::EventSetup& es) const { }
  /// Release any objects obtained from the EventSetup
  virtual void releaseSetup() const { }
};

#endif
