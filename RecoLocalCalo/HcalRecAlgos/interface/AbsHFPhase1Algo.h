#ifndef RecoLocalCalo_HcalRecAlgos_AbsHFPhase1Algo_h_
#define RecoLocalCalo_HcalRecAlgos_AbsHFPhase1Algo_h_

#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HFPreRecHit.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"

class AbsHcalAlgoData;

//
// It is assumed that the HF Phase 1 algorithms will be developed
// utilizing pairs of classes: a DB class with relevant calibration
// constants and configuration parameters, derived from AbsHcalAlgoData,
// and the algo class, derived from AbsHFPhase1Algo, utilizing that DB class.
//
// We can expect that the same calibration constants might be utilized
// by different reco algorithms.
//
// In principle, of course, the configuration objects do not have to be
// actually stored in the database. It is expected that, at the early
// stages of our understanding of energy reconstruction with dual-anode
// PMTs, these objects will be created from the module configuration
// parameters.
//
class AbsHFPhase1Algo {
public:
  inline virtual ~AbsHFPhase1Algo() {}

  // Does this class expect to receive its configuration from the database?
  virtual bool isConfigurable() const = 0;

  // If using DB, expect that the configuration will be updated
  // once per run. We will not manage the pointer here. "true"
  // should be returned on success (typically, automatic cast
  // from the pointer checked by the appropriate dynamic cast).
  inline virtual bool configure(const AbsHcalAlgoData*) { return false; }

  // Convention: if we do not want to use the given HFPreRecHit
  // at all (i.e., it is to be discarded), the returned HFRecHit
  // should have its id (of type HcalDetId) set to 0.
  virtual HFRecHit reconstruct(const HFPreRecHit& prehit,
                               const HcalCalibrations& calibs,
                               const bool flaggedBadInDB[2],
                               bool expectSingleAnodePMT) = 0;
};

#endif  // RecoLocalCalo_HcalRecAlgos_AbsHFPhase1Algo_h_
