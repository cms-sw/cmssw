#include "RecoMET/METPUSubtraction/interface/NoPileUpMEtAuxFunctions.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include <cmath>

namespace noPuUtils{
  const double dR2Min=0.01*0.01;

  int isVertexAssociated(const reco::PFCandidatePtr& pfCandidate,
			 const PFCandToVertexAssMap& pfCandToVertexAssociations,
			 const reco::VertexCollection& vertices, double dZ)
  {
    int vtxAssociationType = noPuUtils::kNeutral;

    if ( pfCandidate->charge() != 0 ) {
      vtxAssociationType = noPuUtils::kChNoAssoc;
      for ( PFCandToVertexAssMap::const_iterator pfCandToVertexAssociation = pfCandToVertexAssociations.begin();
	    pfCandToVertexAssociation != pfCandToVertexAssociations.end(); ++pfCandToVertexAssociation ) {
     
	const noPuUtils::CandQualityPairVector& pfCandidates_vertex = pfCandToVertexAssociation->val;

     
	for ( noPuUtils::CandQualityPairVector::const_iterator pfCandidate_vertex = pfCandidates_vertex.begin();
	      pfCandidate_vertex != pfCandidates_vertex.end(); ++pfCandidate_vertex ) {
	 
	  const reco::PFCandidatePtr pfcVtx= edm::refToPtr(pfCandidate_vertex->first); //<reco::PFCandidatePtr>
	  //std::cout<<pfCandidate<<"   "<<test<<std::endl;

	  if(pfCandidate != pfcVtx ) continue;//std::cout<<" pouet "<<pfCandidate<<"  "<<test<<std::endl;

	  //if(deltaR2(pfCandidate->p4(), pfCandidate_vertex->first->p4()) > dR2Min ) continue;
	  double z = pfCandToVertexAssociation->key->position().z();
	  int quality = pfCandidate_vertex->second;
	  promoteAssocToHSAssoc( quality, z, vertices, dZ, vtxAssociationType, false);
	}
      }
    }

    return vtxAssociationType;
  }

  noPuUtils::reversedPFCandToVertexAssMap 
  reversePFCandToVertexAssociation(const PFCandToVertexAssMap& pfCandToVertexAssociations) {
  
    noPuUtils::reversedPFCandToVertexAssMap revPfCandToVtxAssoc;

    for ( PFCandToVertexAssMap::const_iterator pfCandToVertexAssociation = pfCandToVertexAssociations.begin();
	  pfCandToVertexAssociation != pfCandToVertexAssociations.end(); ++pfCandToVertexAssociation ) {
      const reco::VertexRef& vertex = pfCandToVertexAssociation->key;
  
      const noPuUtils::CandQualityPairVector& pfCandidates_vertex = pfCandToVertexAssociation->val;
      for ( noPuUtils::CandQualityPairVector::const_iterator pfCandidate_vertex = pfCandidates_vertex.begin();
	    pfCandidate_vertex != pfCandidates_vertex.end(); ++pfCandidate_vertex ) {
	revPfCandToVtxAssoc.insert(pfCandidate_vertex->first, std::make_pair(vertex,  pfCandidate_vertex->second));
      }
    }

    return revPfCandToVtxAssoc;
  }

  int 
  isVertexAssociated_fast(const reco::PFCandidateRef& pfCandidate, 
			  const noPuUtils::reversedPFCandToVertexAssMap& pfCandToVertexAssociations,
			  const reco::VertexCollection& vertices, double dZ,
			  int& numWarnings, int maxWarnings) {
    int vtxAssociationType = noPuUtils::kNeutral; 

    if ( pfCandidate->charge() != 0 ) {
      vtxAssociationType = noPuUtils::kChNoAssoc;
    
      const noPuUtils::VertexQualityPairVector* pfCandAssocVtxs = nullptr;
      noPuUtils::reversedPFCandToVertexAssMap::const_iterator itPfcToVtxAss = pfCandToVertexAssociations.find(pfCandidate);
      if ( itPfcToVtxAss != pfCandToVertexAssociations.end() ) {
	pfCandAssocVtxs = &itPfcToVtxAss->val;
      } else {
	for ( noPuUtils::reversedPFCandToVertexAssMap::const_iterator pfcToVtxAssoc = pfCandToVertexAssociations.begin();
	      pfcToVtxAssoc != pfCandToVertexAssociations.end(); ++pfcToVtxAssoc ) {
	  if ( deltaR2(pfCandidate->p4(), pfcToVtxAssoc->key->p4()) < dR2Min ) {
	    pfCandAssocVtxs = &pfcToVtxAssoc->val;
	    break;
	  }
	}
      }
      if ( pfCandAssocVtxs!=nullptr ) {
	for ( noPuUtils::VertexQualityPairVector::const_iterator pfcAssVtx = pfCandAssocVtxs->begin();
	      pfcAssVtx != pfCandAssocVtxs->end(); ++pfcAssVtx ) {
	  double z = pfcAssVtx->first->position().z();
	  int quality = pfcAssVtx->second;
	  promoteAssocToHSAssoc( quality, z, vertices, dZ, vtxAssociationType, false);
	}
      }
    }
  
    return vtxAssociationType;
  }

  void promoteAssocToHSAssoc(int quality, double z, const reco::VertexCollection& vertices,
			     double dZ, int& vtxAssociationType, bool checkdR2) {
  
    if ( quality >= noPuUtils::kChHSAssoc ) {
      for ( reco::VertexCollection::const_iterator vertex = vertices.begin();
	    vertex != vertices.end(); ++vertex ) {
      
	if ( std::abs( z - vertex->position().z()) < dZ ) {
	  if ( vtxAssociationType < noPuUtils::kChHSAssoc ) vtxAssociationType = noPuUtils::kChHSAssoc;
	}
      }
    }

  }

}
