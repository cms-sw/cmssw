#include "RecoMET/METPUSubtraction/interface/NoPileUpMEtAuxFunctions.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include <cmath>

namespace noPuUtils {
  const double dR2Min = 0.01 * 0.01;

  int isVertexAssociated(const reco::PFCandidatePtr& pfCandidate,
                         const PFCandToVertexAssMap& pfCandToVertexAssociations,
                         const reco::VertexCollection& vertices,
                         double dZ) {
    int vtxAssociationType = noPuUtils::kNeutral;

    if (pfCandidate->charge() != 0) {
      vtxAssociationType = noPuUtils::kChNoAssoc;
      for (const auto& pfCandToVertexAssociation : pfCandToVertexAssociations) {
        const noPuUtils::CandQualityPairVector& pfCandidates_vertex = pfCandToVertexAssociation.val;

        for (const auto& pfCandidate_vertex : pfCandidates_vertex) {
          const reco::PFCandidatePtr pfcVtx = edm::refToPtr(pfCandidate_vertex.first);  //<reco::PFCandidatePtr>
          //std::cout<<pfCandidate<<"   "<<test<<std::endl;

          if (pfCandidate != pfcVtx)
            continue;  //std::cout<<" pouet "<<pfCandidate<<"  "<<test<<std::endl;

          //if(deltaR2(pfCandidate->p4(), pfCandidate_vertex->first->p4()) > dR2Min ) continue;
          double z = pfCandToVertexAssociation.key->position().z();
          int quality = pfCandidate_vertex.second;
          promoteAssocToHSAssoc(quality, z, vertices, dZ, vtxAssociationType, false);
        }
      }
    }

    return vtxAssociationType;
  }

  noPuUtils::reversedPFCandToVertexAssMap reversePFCandToVertexAssociation(
      const PFCandToVertexAssMap& pfCandToVertexAssociations) {
    noPuUtils::reversedPFCandToVertexAssMap revPfCandToVtxAssoc;

    for (const auto& pfCandToVertexAssociation : pfCandToVertexAssociations) {
      const reco::VertexRef& vertex = pfCandToVertexAssociation.key;

      const noPuUtils::CandQualityPairVector& pfCandidates_vertex = pfCandToVertexAssociation.val;
      for (const auto& pfCandidate_vertex : pfCandidates_vertex) {
        revPfCandToVtxAssoc.insert(pfCandidate_vertex.first, std::make_pair(vertex, pfCandidate_vertex.second));
      }
    }

    return revPfCandToVtxAssoc;
  }

  int isVertexAssociated_fast(const reco::PFCandidateRef& pfCandidate,
                              const noPuUtils::reversedPFCandToVertexAssMap& pfCandToVertexAssociations,
                              const reco::VertexCollection& vertices,
                              double dZ,
                              int& numWarnings,
                              int maxWarnings) {
    int vtxAssociationType = noPuUtils::kNeutral;

    if (pfCandidate->charge() != 0) {
      vtxAssociationType = noPuUtils::kChNoAssoc;

      const noPuUtils::VertexQualityPairVector* pfCandAssocVtxs = nullptr;
      noPuUtils::reversedPFCandToVertexAssMap::const_iterator itPfcToVtxAss =
          pfCandToVertexAssociations.find(pfCandidate);
      if (itPfcToVtxAss != pfCandToVertexAssociations.end()) {
        pfCandAssocVtxs = &itPfcToVtxAss->val;
      } else {
        for (const auto& pfCandToVertexAssociation : pfCandToVertexAssociations) {
          if (deltaR2(pfCandidate->p4(), pfCandToVertexAssociation.key->p4()) < dR2Min) {
            pfCandAssocVtxs = &pfCandToVertexAssociation.val;
            break;
          }
        }
      }
      if (pfCandAssocVtxs != nullptr) {
        for (const auto& pfCandAssocVtx : *pfCandAssocVtxs) {
          double z = pfCandAssocVtx.first->position().z();
          int quality = pfCandAssocVtx.second;
          promoteAssocToHSAssoc(quality, z, vertices, dZ, vtxAssociationType, false);
        }
      }
    }

    return vtxAssociationType;
  }

  void promoteAssocToHSAssoc(int quality,
                             double z,
                             const reco::VertexCollection& vertices,
                             double dZ,
                             int& vtxAssociationType,
                             bool checkdR2) {
    if (quality >= noPuUtils::kChHSAssoc) {
      for (const auto& vertice : vertices) {
        if (std::abs(z - vertice.position().z()) < dZ) {
          if (vtxAssociationType < noPuUtils::kChHSAssoc)
            vtxAssociationType = noPuUtils::kChHSAssoc;
        }
      }
    }
  }

}  // namespace noPuUtils
