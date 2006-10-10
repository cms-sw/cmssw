#ifndef ImpactParameterAlgorithm_H
#define ImpactParameterAlgorithm_H
#include "DataFormats/BTauReco/interface/TauImpactParameterInfo.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

using namespace std; 
using namespace edm;
using namespace reco;

class  ImpactParameterAlgorithm  {

    public:
 
	ImpactParameterAlgorithm(const ParameterSet  & parameters );
	ImpactParameterAlgorithm(); 

	// For out of framework usage we may need a different constructor
	// so we keep datamember as builtin types (instead of ParameterSet) 
	//ImpactParameterAlgorithm (int,float,....);
   
	~ImpactParameterAlgorithm() {}

	void setPrimaryVertex(Vertex * pv) {primaryVertex = pv;}
  
	pair<JetTag,TauImpactParameterInfo> tag(const IsolatedTauTagInfoRef&, const Vertex&); 

	void setTransientTrackBuilder(const TransientTrackBuilder*);


    private:
	Vertex* primaryVertex;

	//algorithm parameters
	double  ip_min,
		ip_max,
		sip_min;
	bool	use3D,
		use_sign;

	const TransientTrackBuilder * transientTrackBuilder;
};

#endif 

