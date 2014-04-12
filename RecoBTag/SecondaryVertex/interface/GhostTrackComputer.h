#ifndef RecoBTag_SecondaryVertex_GhostTrackComputer_h
#define RecoBTag_SecondaryVertex_GhostTrackComputer_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"  
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

#include "RecoBTag/SecondaryVertex/interface/TrackSelector.h"
#include "RecoBTag/SecondaryVertex/interface/V0Filter.h"

class GhostTrackComputer {
    public:
	GhostTrackComputer(const edm::ParameterSet &params);

	reco::TaggingVariableList
	operator () (const reco::TrackIPTagInfo &ipInfo,
	             const reco::SecondaryVertexTagInfo &svInfo) const;

    private:
	const reco::TrackIPTagInfo::TrackIPData &
	threshTrack(const reco::TrackIPTagInfo &trackIPTagInfo,
	            const reco::TrackIPTagInfo::SortCriteria sort,
	            const reco::Jet &jet,
	            const GlobalPoint &pv) const;

	double					charmCut;
	reco::TrackIPTagInfo::SortCriteria	sortCriterium;
	reco::TrackSelector			trackSelector;
	reco::TrackSelector			trackNoDeltaRSelector;
	double					minTrackWeight;
	bool					vertexMassCorrection;
	reco::V0Filter				trackPairV0Filter;
};

#endif // RecoBTag_SecondaryVertex_GhostTrackComputer_h
