#ifndef RecoMET_METPUSubtraction_noPileUpMEtAuxFunctions_h
#define RecoMET_METPUSubtraction_noPileUpMEtAuxFunctions_h

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "CommonTools/RecoUtils/interface/PFCand_AssoMapAlgos.h"

 // 0 = neutral particle,
 // 1 = charged particle not associated to any vertex
 // 2 = charged particle associated to pile-up vertex
 // 3 = charged particle associated to vertex of hard-scatter event
namespace noPuUtils {
  enum {kNeutral=0, kChNoAssoc, kChPUAssoc, kChHSAssoc};

  typedef std::vector<std::pair<reco::PFCandidateRef, int> > CandQualityPairVector;
  typedef std::vector<std::pair<reco::VertexRef, int> > VertexQualityPairVector;

  typedef edm::AssociationMap<edm::OneToManyWithQuality<reco::PFCandidateCollection, reco::VertexCollection, int> > 
    reversedPFCandToVertexAssMap; 
  
}



int isVertexAssociated(const reco::PFCandidate&, const PFCandToVertexAssMap&, const reco::VertexCollection&, double);

noPuUtils::reversedPFCandToVertexAssMap reversePFCandToVertexAssociation(const PFCandToVertexAssMap&);

int isVertexAssociated_fast(const reco::PFCandidateRef&, const noPuUtils::reversedPFCandToVertexAssMap&, const reco::VertexCollection&, double, int&, int);

void promoteAssocToHSAssoc(int quality, double z,
			   const reco::VertexCollection& vertices,
			   double dZ, int& vtxAssociationType, bool checkdR2);

#endif
