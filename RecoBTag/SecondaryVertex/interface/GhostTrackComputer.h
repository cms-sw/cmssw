#ifndef RecoBTag_SecondaryVertex_GhostTrackComputer_h
#define RecoBTag_SecondaryVertex_GhostTrackComputer_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"  
#include "DataFormats/BTauReco/interface/CandSecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

#include "RecoBTag/SecondaryVertex/interface/TrackSelector.h"
#include "RecoBTag/SecondaryVertex/interface/V0Filter.h"

class GhostTrackComputer {
    public:
	GhostTrackComputer(const edm::ParameterSet &params);
        virtual ~GhostTrackComputer() = default;
	virtual reco::TaggingVariableList
	operator () (const reco::TrackIPTagInfo &ipInfo,
	             const reco::SecondaryVertexTagInfo &svInfo) const;
	virtual reco::TaggingVariableList
	operator () (const reco::CandIPTagInfo &ipInfo,
	             const reco::CandSecondaryVertexTagInfo &svInfo) const;

    private:
	const reco::btag::TrackIPData &
	threshTrack(const reco::TrackIPTagInfo &trackIPTagInfo,
	            const reco::btag::SortCriteria sort,
	            const reco::Jet &jet,
	            const GlobalPoint &pv) const;
	const reco::btag::TrackIPData &
	threshTrack(const reco::CandIPTagInfo &trackIPTagInfo,
	            const reco::btag::SortCriteria sort,
	            const reco::Jet &jet,
	            const GlobalPoint &pv) const;

	double					charmCut;
	reco::btag::SortCriteria		sortCriterium;
	reco::TrackSelector			trackSelector;
	reco::TrackSelector			trackNoDeltaRSelector;
	double					minTrackWeight;
	bool					vertexMassCorrection;
	reco::V0Filter				trackPairV0Filter;
};

#endif // RecoBTag_SecondaryVertex_GhostTrackComputer_h
