#ifndef ImpactParameterAlgorithm_H
#define ImpactParameterAlgorithm_H
#include "DataFormats/BTauReco/interface/TauImpactParameterInfo.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
//#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

 



class  ImpactParameterAlgorithm  {

    public:
 
	ImpactParameterAlgorithm(const edm::ParameterSet  & parameters );
	ImpactParameterAlgorithm(); 

	// For out of framework usage we may need a different constructor
	// so we keep datamember as builtin types (instead of ParameterSet) 
	//ImpactParameterAlgorithm (int,float,....);
   
	~ImpactParameterAlgorithm() {}

	void setPrimaryVertex(reco::Vertex * pv) {primaryVertex = pv;}
  
	std::pair<float,reco::TauImpactParameterInfo> tag(const reco::IsolatedTauTagInfoRef&, const reco::Vertex&); 

	void setTransientTrackBuilder(const TransientTrackBuilder*);


    private:
	reco::Vertex* primaryVertex;

	//algorithm parameters
	double  ip_min,
		ip_max,
		sip_min;
	bool	use3D,
		use_sign;

	const TransientTrackBuilder * transientTrackBuilder;
};

#endif 

