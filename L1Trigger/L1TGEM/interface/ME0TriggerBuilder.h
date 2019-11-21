#ifndef L1Trigger_L1TGEM_ME0TriggerBuilder_h
#define L1Trigger_L1TGEM_ME0TriggerBuilder_h

/** \class ME0TriggerBuilder
 *
 * Builds ME0 trigger objects from ME0 pads
 *
 * \author Sven Dildick (TAMU)
 *
 */

#include "DataFormats/GEMDigi/interface/ME0TriggerDigiCollection.h"
#include "DataFormats/GEMDigi/interface/ME0PadDigiCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1TGEM/interface/ME0Motherboard.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class ME0TriggerBuilder {
public:
  /** Configure the algorithm via constructor.
   *  Receives ParameterSet percolated down from
   *  EDProducer which owns this Builder.
   */
  explicit ME0TriggerBuilder(const edm::ParameterSet&);

  ~ME0TriggerBuilder();

  /** Build Triggers from pads in each chamber and fill them into output collections. */
  void build(const ME0PadDigiCollection* me0Pads, ME0TriggerDigiCollection& oc_trig);

  /** set geometry for the matching needs */
  void setME0Geometry(const ME0Geometry* g) { me0_g = g; }

  /** Max values of trigger labels for all ME0s;
   *  used to construct TMB processors.
   */
  enum trig_me0s { MAX_ENDCAPS = 2, MAX_CHAMBERS = 18 };

private:
  static const int min_endcap;
  static const int max_endcap;
  static const int min_chamber;
  static const int max_chamber;

  const ME0Geometry* me0_g;

  edm::ParameterSet config_;

  /** Pointers to TMB processors for all possible chambers. */
  std::unique_ptr<ME0Motherboard> tmb_[MAX_ENDCAPS][MAX_CHAMBERS];
};

#endif
