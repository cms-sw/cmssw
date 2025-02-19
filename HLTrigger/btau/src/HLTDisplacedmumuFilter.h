#ifndef HLTDisplacedmumuFilter_h
#define HLTDisplacedmumuFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
	
class HLTDisplacedmumuFilter : public HLTFilter {
 public:
		explicit HLTDisplacedmumuFilter(const edm::ParameterSet&);
		~HLTDisplacedmumuFilter();
	
		virtual void beginJob() ;
		virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);
		virtual void endJob() ;
		
 private:

		bool fastAccept_;
		double minLxySignificance_;
		double maxLxySignificance_;
		double maxNormalisedChi2_;
		double minVtxProbability_;
		double minCosinePointingAngle_;
		edm::InputTag DisplacedVertexTag_;
		edm::InputTag beamSpotTag_;
		edm::InputTag MuonTag_;
};
#endif
