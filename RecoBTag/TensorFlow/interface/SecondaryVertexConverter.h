#ifndef RecoBTag_TensorFlow_SecondaryVertexConverter_h
#define RecoBTag_TensorFlow_SecondaryVertexConverter_h

#include "RecoBTag/TensorFlow/interface/deep_helpers.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexFeatures.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

namespace btagbtvdeep {
  
  
  void svToFeatures( const reco::VertexCompositePtrCandidate & sv,
		     const reco::Vertex & pv, const reco::Jet & jet,
		     SecondaryVertexFeatures & sv_features,
		     const bool flip = false) ;
    
  
}

#endif //RecoSV_DeepFlavour_SecondaryVertexConverter_h
