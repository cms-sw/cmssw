#ifndef L1Trigger_L1TGEM_ME0StubBuilderV2_H
#define L1Trigger_L1TGEM_ME0StubBuilderV2_H

/** \class ME0StubBuilderV2 derived by GEMSegmentBuilder
 * Algorithm to build ME0Stub's from GEMPadDigi collection
 * by implementing a 'build' function required by ME0StubProducer.
 *
 * Implementation notes: <BR>
 * Configured via the Producer's ParameterSet. <BR>
 * Presume this might become an abstract base class one day. <BR>
 *
 * \author Woohyeon Heo
 *
*/
#include <cstdint>

#include "DataFormats/GEMDigi/interface/ME0TriggerDigiCollection.h"
#include "DataFormats/GEMDigi/interface/ME0TriggerDigi.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "L1Trigger/L1TGEM/interface/ME0StubPrimitive.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class ME0StubBuilderV2 {
public:
  explicit ME0StubBuilderV2(const edm::ParameterSet&);
  ~ME0StubBuilderV2();

  void build(const GEMPadDigiCollection* padDigis, GE0TriggerDigiCollection& oc);

  static void fillDescription(edm::ParameterSetDescription& descriptions);

  void setME0Geometry(const GEMGeometry* g) { me0Geom_ = g; }

  ME0TriggerDigi stubConversion(const ME0StubPrimitive& stub, const GEMDetId& id, int crntBX);

private:
  const GEMGeometry* me0Geom_;

  bool skipCentroids_;
  std::vector<int32_t> layerThresholdPatternId_;
  std::vector<int32_t> layerThresholdEta_;
  int32_t maxSpan_;
  int32_t width_;
  bool deghostPre_;
  bool deghostPost_;
  int32_t groupWidth_;
  int32_t ghostWidth_;
  bool xPartitionEnabled_;
  bool enableNonPointing_;
  int32_t crossPartitionSegmentWidth_;
  int32_t clearanceWidth_;
  int32_t numOutputs_;
  bool checkIds_;
  int32_t edgeDistance_;
  int32_t numOr_;
  double mseThreshold_;
  double bendAngleCut_;
  int32_t BXWindow_;
  bool debug_ = false;
};

#endif
