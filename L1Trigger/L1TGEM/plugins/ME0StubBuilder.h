#ifndef L1Trigger_L1TGEM_ME0StubBuilder_h
#define L1Trigger_L1TGEM_ME0StubBuilder_h

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

  void build(const GEMPadDigiCollection* paddigis, ME0StubCollection& oc);

  static void fillDescription(edm::ParameterSetDescription& descriptions);

private:
  bool skip_centroids;
  std::vector<int32_t> ly_thresh_patid;
  std::vector<int32_t> ly_thresh_eta;
  int32_t max_span;
  int32_t width;
  bool deghost_pre;
  bool deghost_post;
  int32_t group_width;
  int32_t ghost_width;
  bool x_prt_en;
  bool en_non_pointing;
  int32_t cross_part_seg_width;
  int32_t num_outputs;
  bool check_ids;
  int32_t edge_distance;
  int32_t num_or;
  double mse_thresh;
};

#endif
