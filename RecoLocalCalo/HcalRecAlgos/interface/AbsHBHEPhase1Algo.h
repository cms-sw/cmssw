#ifndef RecoLocalCalo_HcalRecAlgos_AbsHBHEPhase1Algo_h_
#define RecoLocalCalo_HcalRecAlgos_AbsHBHEPhase1Algo_h_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHEChannelInfo.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParam.h"

class AbsHcalAlgoData;

//
// It is assumed that the HBHE Phase 1 algorithms will be developed
// utilizing pairs of classes: a DB class with relevant calibration
// constants and configuration parameters, derived from AbsHcalAlgoData,
// and the algo class, derived from AbsHBHEPhase1Algo, utilizing that DB class.
//
// We can expect that the same calibration constants might be utilized
// by different reco algorithms.
//
// In principle, of course, the configuration objects do not have to be
// actually stored in the database. It is expected that, at the early
// stages of our understanding of energy reconstruction with QIE11
// ASICs, these objects will be created from the module configuration
// parameters.
//
class AbsHBHEPhase1Algo {
public:
  inline virtual ~AbsHBHEPhase1Algo() {}

  inline virtual void beginRun(const edm::Run&, const edm::EventSetup&) {}
  inline virtual void endRun() {}

  // Does this class expect to receive its configuration from the database?
  virtual bool isConfigurable() const = 0;

  // If using DB, expect that the configuration will be updated
  // once per run. We will not manage the pointer here. "true"
  // should be returned on success (typically, automatic cast
  // from the pointer checked by the appropriate dynamic cast).
  inline virtual bool configure(const AbsHcalAlgoData*) { return false; }

  // Convention: if we do not want to use the given channel at
  // all (i.e., it is to be discarded), the returned HBHERecHit
  // should have its id (of type HcalDetId) set to 0.
  //
  // Note that "params" pointer is allowed to be null.
  //
  virtual HBHERecHit reconstruct(const HBHEChannelInfo& info,
                                 const HcalRecoParam* params,
                                 const HcalCalibrations& calibs,
                                 bool isRealData) = 0;
};

#endif  // RecoLocalCalo_HcalRecAlgos_AbsHBHEPhase1Algo_h_
