#include "RecoMET/METPUSubtraction/interface/noPileUpMEtAuxFunctions.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <math.h>

const int minPFCandToVertexAssocQuality = noPuUtils::kChHSAssoc; // CV: value recommended by Matthias Geisler, representing "good" PFCandidate-vertex associations

const double dRMin=0.01;

int isVertexAssociated(const reco::PFCandidate& pfCandidate,
                       const PFCandToVertexAssMap& pfCandToVertexAssociations,
                       const reco::VertexCollection& vertices, double dZ)
{
  int vtxAssociationType = noPuUtils::kNeutral;

  if ( pfCandidate.charge() != 0 ) {
    vtxAssociationType = noPuUtils::kChNoAssoc;
    for ( PFCandToVertexAssMap::const_iterator pfCandToVertexAssociation = pfCandToVertexAssociations.begin();
          pfCandToVertexAssociation != pfCandToVertexAssociations.end(); ++pfCandToVertexAssociation ) {
      typedef std::vector<std::pair<reco::PFCandidateRef, int> > PFCandidateQualityPairVector;
      const PFCandidateQualityPairVector& pfCandidates_vertex = pfCandToVertexAssociation->val;
      for ( PFCandidateQualityPairVector::const_iterator pfCandidate_vertex = pfCandidates_vertex.begin();
            pfCandidate_vertex != pfCandidates_vertex.end(); ++pfCandidate_vertex ) {
	int pfCandToVertexAssocQuality = pfCandidate_vertex->second;
	if ( pfCandToVertexAssocQuality >= minPFCandToVertexAssocQuality && deltaR2(pfCandidate.p4(), pfCandidate_vertex->first->p4()) < dRMin*dRMin ) {
          if ( vtxAssociationType < noPuUtils::kChPUAssoc ) vtxAssociationType = noPuUtils::kChPUAssoc;
          for ( reco::VertexCollection::const_iterator vertex = vertices.begin();
                vertex != vertices.end(); ++vertex ) {
            if ( fabs(pfCandToVertexAssociation->key->position().z() - vertex->position().z()) < dZ ) {
              if ( vtxAssociationType < noPuUtils::kChHSAssoc ) vtxAssociationType = noPuUtils::kChHSAssoc;
            }
          }
        }
      }
    }
  }

  return vtxAssociationType;
}

reversedPFCandidateToVertexAssociationMap reversePFCandToVertexAssociation(const PFCandToVertexAssMap& pfCandToVertexAssociations)
{
  reversedPFCandidateToVertexAssociationMap pfCandToVertexAssociations_reversed;

  for ( PFCandToVertexAssMap::const_iterator pfCandToVertexAssociation = pfCandToVertexAssociations.begin();
	pfCandToVertexAssociation != pfCandToVertexAssociations.end(); ++pfCandToVertexAssociation ) {
    const reco::VertexRef& vertex = pfCandToVertexAssociation->key;
    typedef std::vector<std::pair<reco::PFCandidateRef, int> > PFCandidateQualityPairVector;
    const PFCandidateQualityPairVector& pfCandidates_vertex = pfCandToVertexAssociation->val;
    for ( PFCandidateQualityPairVector::const_iterator pfCandidate_vertex = pfCandidates_vertex.begin();
	  pfCandidate_vertex != pfCandidates_vertex.end(); ++pfCandidate_vertex ) {
      pfCandToVertexAssociations_reversed.insert(pfCandidate_vertex->first, std::make_pair(vertex,  pfCandidate_vertex->second));
    }
  }

  return pfCandToVertexAssociations_reversed;
}

int isVertexAssociated_fast(const reco::PFCandidateRef& pfCandidate, 
			    const reversedPFCandidateToVertexAssociationMap& pfCandToVertexAssociations,
			    const reco::VertexCollection& vertices, double dZ,
			    int& numWarnings, int maxWarnings)
{
  int vtxAssociationType = 0; // 0 = neutral particle, 
                              // 1 = charged particle not associated to any vertex
                              // 2 = charged particle associated to pile-up vertex
                              // 3 = charged particle associated to vertex of hard-scatter event

  if ( pfCandidate->charge() != 0 ) {
    vtxAssociationType = 1;
    typedef std::vector<std::pair<reco::VertexRef, int> > VertexQualityPairVector;
    const VertexQualityPairVector* pfCandidate_associated_vertices = nullptr;
    reversedPFCandidateToVertexAssociationMap::const_iterator pfCandToVertexAssociation_iter = pfCandToVertexAssociations.find(pfCandidate);
    if ( pfCandToVertexAssociation_iter != pfCandToVertexAssociations.end() ) {
      pfCandidate_associated_vertices = &pfCandToVertexAssociation_iter->val;
    } else {
      for ( reversedPFCandidateToVertexAssociationMap::const_iterator pfCandToVertexAssociation = pfCandToVertexAssociations.begin();
	    pfCandToVertexAssociation != pfCandToVertexAssociations.end(); ++pfCandToVertexAssociation ) {
	if ( deltaR2(pfCandidate->p4(), pfCandToVertexAssociation->key->p4()) < dRMin*dRMin ) {
    	  pfCandidate_associated_vertices = &pfCandToVertexAssociation->val;
    	  break;
    	}
      }
      if ( numWarnings < maxWarnings ) {
    	edm::LogWarning ("isVertexAssociated") 
    	  << " The productIDs of PFCandidate and PFCandToVertexAssociationMap passed as function arguments don't match.\n" 
    	  << "NOTE: The return value will be unaffected, but the code will run MUCH slower !!";
    	++numWarnings;
      }
    }
    if ( pfCandidate_associated_vertices ) {
      for ( VertexQualityPairVector::const_iterator pfCandidate_associated_vertex = pfCandidate_associated_vertices->begin();
	    pfCandidate_associated_vertex != pfCandidate_associated_vertices->end(); ++pfCandidate_associated_vertex ) {	
	if ( pfCandidate_associated_vertex->second >= minPFCandToVertexAssocQuality ) {
	  if ( vtxAssociationType <  noPuUtils::kChPUAssoc ) vtxAssociationType =  noPuUtils::kChPUAssoc;
	  for ( reco::VertexCollection::const_iterator vertex = vertices.begin();
		vertex != vertices.end(); ++vertex ) {
	    if ( fabs(pfCandidate_associated_vertex->first->position().z() - vertex->position().z()) < dZ ) {
	      if ( vtxAssociationType < noPuUtils::kChHSAssoc ) vtxAssociationType = noPuUtils::kChHSAssoc;
	    }
	  }
	}
      }
    }
  }

  return vtxAssociationType;
}
