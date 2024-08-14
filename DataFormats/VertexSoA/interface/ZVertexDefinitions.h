#ifndef DataFormats_VertexSoA_ZVertexDefinitions_h
#define DataFormats_VertexSoA_ZVertexDefinitions_h

#include <cstdint>

namespace zVertex {

  constexpr uint32_t MAXTRACKS = 32 * 1024;
  constexpr uint32_t MAXVTX = 1024;

  //
  // FIXME: MAXTRACKS is low for Phase2 pixel triplets with PU=200
  // and the txiplets wfs in those conditions will fail.
  //
  // Not rising it since to the needed 128*1024 since it causes
  // a 4-5% jump in memory usage.
  //
  // The original sin is that, for the moment, the VertexSoA
  // has a unique index for the tracks and the vertices and
  // so it needs to be sized for the bigger of the two (the tracks, of course).
  // This means we are wasting memory (this is true also for Phase1).
  // We need to split the two indices (as is for CUDA, since we were not using
  // the PortableCollection + Layout was not yet used).
  //

}  // namespace zVertex

#endif
