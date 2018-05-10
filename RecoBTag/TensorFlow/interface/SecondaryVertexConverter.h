#ifndef RecoBTag_DeepFlavour_SecondaryVertexConverter_h
#define RecoBTag_DeepFlavour_SecondaryVertexConverter_h

#include "RecoBTag/DeepFlavour/interface/deep_helpers.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexFeatures.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

namespace btagbtvdeep {
  
  
  void sVToFeatures( const reco::VertexCompositePtrCandidate & sv,
		     const reco::Vertex & pv, const reco::Jet & jet,
		     SecondaryVertexFeatures & sv_features) ;
    
  
}

#endif //RecoSV_DeepFlavour_SecondaryVertexConverter_h
