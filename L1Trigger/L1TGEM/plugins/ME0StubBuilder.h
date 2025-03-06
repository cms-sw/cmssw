#ifndef L1Trigger_L1TGEM_ME0StubBuilder_H
#define L1Trigger_L1TGEM_ME0StubBuilder_H

/** \class ME0StubBuilder derived by GEMSegmentBuilder
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

#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "L1Trigger/L1TGEM/interface/ME0StubPrimitive.h"
#include "DataFormats/GEMDigi/interface/ME0StubCollection.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class ME0StubAlgorithmBase;

class ME0StubBuilder {
public:
  explicit ME0StubBuilder(const edm::ParameterSet&);
  ~ME0StubBuilder();

  void build(const GEMPadDigiCollection* padDigis, ME0StubCollection& oc);

  static void fillDescription(edm::ParameterSetDescription& descriptions);

private:
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
  int32_t numOutputs_;
  bool checkIds_;
  int32_t edgeDistance_;
  int32_t numOr_;
  double mseThreshold_;
};

#endif
