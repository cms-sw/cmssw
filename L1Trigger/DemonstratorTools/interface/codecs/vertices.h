
#ifndef L1Trigger_DemonstratorTools_codecs_vertices_h
#define L1Trigger_DemonstratorTools_codecs_vertices_h

#include <array>
#include <vector>

#include "ap_int.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/L1Trigger/interface/Vertex.h"
#include "DataFormats/L1Trigger/interface/VertexWord.h"

namespace l1t::demo::codecs {

  ap_uint<64> encodeVertex(const l1t::VertexWord& v);

  // Encodes vertex collection onto 1 'logical' output link
  std::array<std::vector<ap_uint<64>>, 1> encodeVertices(const edm::View<l1t::VertexWord>&);

  std::vector<l1t::VertexWord> decodeVertices(const std::vector<ap_uint<64>>&);

}  // namespace l1t::demo::codecs

#endif
