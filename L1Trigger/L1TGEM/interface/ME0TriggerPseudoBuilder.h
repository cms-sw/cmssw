#ifndef L1Trigger_L1TGEM_ME0TriggerPseudoBuilder_h
#define L1Trigger_L1TGEM_ME0TriggerPseudoBuilder_h

/** \class ME0TriggerPseudoBuilder
 *
 * Builds ME0 trigger objects from ME0 segment
 *
 * \author Tao Huang (TAMU)
 *
 */

#include "DataFormats/GEMDigi/interface/ME0TriggerDigiCollection.h"
#include "DataFormats/GEMRecHit/interface/ME0SegmentCollection.h"
#include "DataFormats/GEMRecHit/interface/ME0Segment.h"
#include "DataFormats/GEMRecHit/interface/ME0RecHit.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ME0Geometry;

class ME0TriggerPseudoBuilder {
public:
  /** Configure the algorithm via constructor.
   *  Receives ParameterSet percolated down from
   *  EDProducer which owns this Builder.
   */
  explicit ME0TriggerPseudoBuilder(const edm::ParameterSet&);

  ~ME0TriggerPseudoBuilder();

  /** Build Triggers from ME0 segment in each chamber and fill them into output collections. */
  void build(const ME0SegmentCollection* me0segments, ME0TriggerDigiCollection& oc_trig);

  /** set geometry for the matching needs */
  void setME0Geometry(const ME0Geometry* g) { me0_g = g; }

  /* print all ME0 segments in the event */
  void dumpAllME0Segments(const ME0SegmentCollection& segments) const;

  /** Max values of trigger labels for all ME0s;
   *  used to construct TMB processors.
   */
  enum class trig_me0s { MAX_ENDCAPS = 2, MAX_CHAMBERS = 18 };

private:
  static const int min_endcap;
  static const int max_endcap;
  static const int min_chamber;
  static const int max_chamber;
  static const unsigned int ME0KeyLayer;
  static const int ME0TriggerCentralBX;

  const ME0Geometry* me0_g;

  int info_;

  double dphiresolution_;  //unit: trigger pad

  ME0TriggerDigi segmentConversion(const ME0Segment segment);

  edm::ParameterSet config_;
};

#endif
