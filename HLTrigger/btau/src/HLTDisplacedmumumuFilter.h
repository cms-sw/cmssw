#ifndef HLTDisplacedmumumuFilter_h
#define HLTDisplacedmumumuFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
	
class HLTDisplacedmumumuFilter : public HLTFilter {
 public:
		explicit HLTDisplacedmumumuFilter(const edm::ParameterSet&);
		~HLTDisplacedmumumuFilter();
	
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
