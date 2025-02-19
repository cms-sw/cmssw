#ifndef CALIBFORMATS_CALOTPG_ECALTPGSCALE_H
#define CALIBFORMATS_CALOTPG_ECALTPGSCALE_H 1

#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h"

namespace edm {
  class EventSetup;
}

/** \class EcalTPGScale
 *
 *  To be stored in the CaloTPGRecord
 *  
 * $Date: 2007/02/19 16:24:20 $
 * $Revision: 1.1 $
 * \author J. Mans - Minnesota
 */
class EcalTPGScale {
public:
  /** \brief nominal ET value of this sample/code as to be used in
      RCT LUT creation */
  virtual double et_RCT(const EcalTrigTowerDetId& id, const
			EcalTriggerPrimitiveSample& s) const = 0;
  /// smallest ET value which would resolve into this sample
  virtual double et_bin_low(const EcalTrigTowerDetId& id, const
			    EcalTriggerPrimitiveSample& s) const = 0;
  /** smallest ET value which would not resolve into sample (too big) */
  virtual double et_bin_high(const EcalTrigTowerDetId& id, const
		     EcalTriggerPrimitiveSample& s) const = 0;
  /// Get any needed information from the event setup
  virtual void setup(const edm::EventSetup& es) const { }
  /// Release any objects obtained from the EventSetup
  virtual void releaseSetup() const { }
};

#endif
