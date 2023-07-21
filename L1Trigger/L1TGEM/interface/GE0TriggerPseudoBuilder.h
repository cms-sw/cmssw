#ifndef L1Trigger_L1TGEM_GE0TriggerPseudoBuilder_h
#define L1Trigger_L1TGEM_GE0TriggerPseudoBuilder_h

/** \class GE0TriggerPseudoBuilder
 *
 * Builds GE0 trigger objects from GE0 segment
 *
 * \author Original ME0 code by Tao Huang (TAMU). Converted and updated to GE0 by Ian J. Watson (USeoul)
 *
 */

#include "DataFormats/GEMDigi/interface/ME0TriggerDigiCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMSegmentCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMSegment.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class GEMGeometry;

class GE0TriggerPseudoBuilder {
public:
  /** Configure the algorithm via constructor.
   *  Receives ParameterSet percolated down from
   *  EDProducer which owns this Builder.
   */
  explicit GE0TriggerPseudoBuilder(const edm::ParameterSet&);

  ~GE0TriggerPseudoBuilder();

  /** Build Triggers from ME0 segment in each chamber and fill them into output collections. */
  void build(const GEMSegmentCollection* me0segments, GE0TriggerDigiCollection& oc_trig);

  /** set geometry for the matching needs */
  void setME0Geometry(const GEMGeometry* g) { me0_g = g; }

  /* print all ME0 segments in the event */
  void dumpAllME0Segments(const GEMSegmentCollection& segments) const;

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

  const GEMGeometry* me0_g;

  int info_;

  double dphiresolution_;  //unit: trigger pad

  ME0TriggerDigi segmentConversion(const GEMSegment segment);

  edm::ParameterSet config_;
};

#endif
