#ifndef RecoBTag_SecondaryVertex_CombinedSVComputerV2_h
#define RecoBTag_SecondaryVertex_CombinedSVComputerV2_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"  
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

#include "RecoBTag/SecondaryVertex/interface/TrackSelector.h"
#include "RecoBTag/SecondaryVertex/interface/V0Filter.h"

class CombinedSVComputerV2 {
    public:
	CombinedSVComputerV2(const edm::ParameterSet &params);

	reco::TaggingVariableList
	operator () (const reco::TrackIPTagInfo &ipInfo,
	             const reco::SecondaryVertexTagInfo &svInfo) const;

    private:
	struct IterationRange;

	double flipValue(double value, bool vertex) const;
	IterationRange flipIterate(int size, bool vertex) const;

	const reco::TrackIPTagInfo::TrackIPData &
	threshTrack(const reco::TrackIPTagInfo &trackIPTagInfo,
	            const reco::TrackIPTagInfo::SortCriteria sort,
	            const reco::Jet &jet,
	            const GlobalPoint &pv) const;

	bool					trackFlip;
	bool					vertexFlip;
	double					charmCut;
	reco::TrackIPTagInfo::SortCriteria	sortCriterium;
	reco::TrackSelector			trackSelector;
	reco::TrackSelector			trackNoDeltaRSelector;
	reco::TrackSelector			trackPseudoSelector;
	unsigned int				pseudoMultiplicityMin;
	unsigned int				trackMultiplicityMin;
	double					minTrackWeight;
	bool					useTrackWeights;
	bool					vertexMassCorrection;
	reco::V0Filter				pseudoVertexV0Filter;
	reco::V0Filter				trackPairV0Filter;
};

#endif // RecoBTag_SecondaryVertex_CombinedSVComputerV2_h
