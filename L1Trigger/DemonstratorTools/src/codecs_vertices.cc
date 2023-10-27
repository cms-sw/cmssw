
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
      VertexWord::vtxvalid_t validWord;
      VertexWord::vtxz0_t z0Word;
      VertexWord::vtxmultiplicity_t multWord;
      VertexWord::vtxsumpt_t sumPtWord;
      VertexWord::vtxquality_t qualityWord;
      VertexWord::vtxinversemult_t inverseMultWord;
      VertexWord::vtxunassigned_t unassignedWord;

      validWord.V = x(VertexWord::kValidMSB, VertexWord::kValidLSB);
      z0Word.V = x(VertexWord::kZ0MSB, VertexWord::kZ0LSB);
      multWord.V = x(VertexWord::kNTrackInPVMSB, VertexWord::kNTrackInPVLSB);
      sumPtWord.V = x(VertexWord::kSumPtMSB, VertexWord::kSumPtLSB);
      qualityWord.V = x(VertexWord::kQualityMSB, VertexWord::kQualityLSB);
      inverseMultWord.V = x(VertexWord::kNTrackOutPVMSB, VertexWord::kNTrackOutPVLSB);
      unassignedWord.V = x(VertexWord::kUnassignedMSB, VertexWord::kUnassignedLSB);

      VertexWord v(validWord,
                   z0Word,
                   multWord,
                   sumPtWord,
                   qualityWord,
                   qualityWord,
                   unassignedWord);
      vertices.push_back(v);
    }

    return vertices;
  }

}  // namespace l1t::demo::codecs
