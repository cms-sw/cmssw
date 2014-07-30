#include "RecoMET/METPUSubtraction/interface/noPileUpMEtAuxFunctions.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <math.h>

const int minPFCandToVertexAssocQuality = 3; // CV: value recommended by Matthias Geisler, representing "good" PFCandidate-vertex associations

int isVertexAssociated(const reco::PFCandidate& pfCandidate,
                       const PFCandToVertexAssMap& pfCandToVertexAssociations,
                       const reco::VertexCollection& vertices, double dZ)
{
  int vtxAssociationType = 0; // 0 = neutral particle,
                              // 1 = charged particle not associated to any vertex
                              // 2 = charged particle associated to pile-up vertex
                              // 3 = charged particle associated to vertex of hard-scatter event

  if ( fabs(pfCandidate.charge()) > 0.5 ) {
    vtxAssociationType = 1;
    for ( PFCandToVertexAssMap::const_iterator pfCandToVertexAssociation = pfCandToVertexAssociations.begin();
          pfCandToVertexAssociation != pfCandToVertexAssociations.end(); ++pfCandToVertexAssociation ) {
      typedef std::vector<std::pair<reco::PFCandidateRef, int> > PFCandidateQualityPairVector;
      const PFCandidateQualityPairVector& pfCandidates_vertex = pfCandToVertexAssociation->val;
      for ( PFCandidateQualityPairVector::const_iterator pfCandidate_vertex = pfCandidates_vertex.begin();
            pfCandidate_vertex != pfCandidates_vertex.end(); ++pfCandidate_vertex ) {
	int pfCandToVertexAssocQuality = pfCandidate_vertex->second;
	if ( pfCandToVertexAssocQuality >= minPFCandToVertexAssocQuality && deltaR(pfCandidate.p4(), pfCandidate_vertex->first->p4()) < 1.e-2 ) {
          if ( vtxAssociationType < 2 ) vtxAssociationType = 2;
          for ( reco::VertexCollection::const_iterator vertex = vertices.begin();
                vertex != vertices.end(); ++vertex ) {
            if ( fabs(pfCandToVertexAssociation->key->position().z() - vertex->position().z()) < dZ ) {
              if ( vtxAssociationType < 3 ) vtxAssociationType = 3;
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

  if ( fabs(pfCandidate->charge()) > 0.5 ) {
    vtxAssociationType = 1;
    typedef std::vector<std::pair<reco::VertexRef, int> > VertexQualityPairVector;
    const VertexQualityPairVector* pfCandidate_associated_vertices = 0;
    reversedPFCandidateToVertexAssociationMap::const_iterator pfCandToVertexAssociation_iter = pfCandToVertexAssociations.find(pfCandidate);
    if ( pfCandToVertexAssociation_iter != pfCandToVertexAssociations.end() ) {
      pfCandidate_associated_vertices = &pfCandToVertexAssociation_iter->val;
    } else {
      for ( reversedPFCandidateToVertexAssociationMap::const_iterator pfCandToVertexAssociation = pfCandToVertexAssociations.begin();
	    pfCandToVertexAssociation != pfCandToVertexAssociations.end(); ++pfCandToVertexAssociation ) {
	if ( deltaR(pfCandidate->p4(), pfCandToVertexAssociation->key->p4()) < 1.e-2 ) {
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
	  if ( vtxAssociationType < 2 ) vtxAssociationType = 2;
	  for ( reco::VertexCollection::const_iterator vertex = vertices.begin();
		vertex != vertices.end(); ++vertex ) {
	    if ( fabs(pfCandidate_associated_vertex->first->position().z() - vertex->position().z()) < dZ ) {
	      if ( vtxAssociationType < 3 ) vtxAssociationType = 3;
	    }
	  }
	}
      }
    }
  }

  return vtxAssociationType;
}
