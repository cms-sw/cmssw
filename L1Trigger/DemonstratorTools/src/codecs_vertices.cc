
#include "L1Trigger/DemonstratorTools/interface/codecs/vertices.h"

namespace l1t::demo::codecs {

  ap_uint<64> encodeVertex(const l1t::VertexWord& v) { return v.vertexWord(); }

  // Encodes vertex collection onto 1 output link
  std::array<std::vector<ap_uint<64>>, 1> encodeVertices(const edm::View<l1t::VertexWord>& vertices) {
    std::vector<ap_uint<64>> vertexWords;

    for (const auto& vertex : vertices)
      vertexWords.push_back(encodeVertex(vertex));

    std::array<std::vector<ap_uint<64>>, 1> linkData;

    for (size_t i = 0; i < linkData.size(); i++) {
      // Pad vertex vectors -> full packet length (48 frames, but only 10 vertices max)
      vertexWords.resize(10, 0);
      linkData.at(i) = vertexWords;
    }

    return linkData;
  }

  std::vector<l1t::VertexWord> decodeVertices(const std::vector<ap_uint<64>>& frames) {
    std::vector<l1t::VertexWord> vertices;

    for (const auto& x : frames) {
      if (not x.test(VertexWord::kValidLSB))
        break;

      VertexWord v(VertexWord::vtxvalid_t(1),
                   VertexWord::vtxz0_t(x(VertexWord::kZ0MSB, VertexWord::kZ0LSB)),
                   VertexWord::vtxmultiplicity_t(x(VertexWord::kNTrackInPVMSB, VertexWord::kNTrackInPVLSB)),
                   VertexWord::vtxsumpt_t(x(VertexWord::kSumPtMSB, VertexWord::kSumPtLSB)),
                   VertexWord::vtxquality_t(x(VertexWord::kQualityMSB, VertexWord::kQualityLSB)),
                   VertexWord::vtxinversemult_t(x(VertexWord::kNTrackOutPVMSB, VertexWord::kNTrackOutPVLSB)),
                   VertexWord::vtxunassigned_t(x(VertexWord::kUnassignedMSB, VertexWord::kUnassignedLSB)));
      vertices.push_back(v);
    }

    return vertices;
  }

}  // namespace l1t::demo::codecs
