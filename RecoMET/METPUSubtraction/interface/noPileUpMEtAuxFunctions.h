#ifndef RecoMET_METPUSubtraction_noPileUpMEtAuxFunctions_h
#define RecoMET_METPUSubtraction_noPileUpMEtAuxFunctions_h

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "CommonTools/RecoUtils/interface/PFCand_AssoMapAlgos.h"

typedef edm::AssociationMap<edm::OneToManyWithQuality<reco::PFCandidateCollection, reco::VertexCollection, int> >
  reversedPFCandidateToVertexAssociationMap;

int isVertexAssociated(const reco::PFCandidate&, const PFCandToVertexAssMap&, const reco::VertexCollection&, double);

reversedPFCandidateToVertexAssociationMap reversePFCandToVertexAssociation(const PFCandToVertexAssMap&);

int isVertexAssociated_fast(const reco::PFCandidateRef&, const reversedPFCandidateToVertexAssociationMap&, const reco::VertexCollection&, double, int&, int);

#endif
