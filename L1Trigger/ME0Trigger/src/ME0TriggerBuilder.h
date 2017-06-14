#ifndef L1Trigger_ME0Trigger_ME0TriggerBuilder_h
#define L1Trigger_ME0Trigger_ME0TriggerBuilder_h

/** \class ME0TriggerBuilder
 *
 * Builds ME0 trigger objects from ME0 pad clusters
 *
 * \author Sven Dildick (TAMU)
 *
 */

#include "DataFormats/GEMDigi/interface/ME0TriggerDigiCollection.h"
#include "DataFormats/GEMDigi/interface/ME0PadDigiClusterCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ME0Motherboard;
class ME0Geometry;

class ME0TriggerBuilder
{
 public:

  /** Configure the algorithm via constructor.
   *  Receives ParameterSet percolated down from 
   *  EDProducer which owns this Builder.
   */
  explicit ME0TriggerBuilder(const edm::ParameterSet&);

  ~ME0TriggerBuilder();

  /** Build Triggers from clusters in each chamber and fill them into output collections. */
  void build(const ME0PadDigiClusterCollection* me0Pads, ME0TriggerDigiCollection& oc_trig);

  /** set geometry for the matching needs */
  void setME0Geometry(const ME0Geometry *g) { me0_g = g; }

  /** Max values of trigger labels for all ME0s; 
   *  used to construct TMB processors. 
   */
  enum trig_me0s {MAX_ENDCAPS = 2, MAX_CHAMBERS = 18};

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
