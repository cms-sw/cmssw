#ifndef RecoBTag_SecondaryVertex_CombinedSVComputer_h
#define RecoBTag_SecondaryVertex_CombinedSVComputer_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"  
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

#include "RecoBTag/SecondaryVertex/interface/TrackSelector.h"

class CombinedSVComputer {
    public:
	CombinedSVComputer(const edm::ParameterSet &params);

	reco::TaggingVariableList
	operator () (const reco::TrackIPTagInfo &ipInfo,
	             const reco::SecondaryVertexTagInfo &svInfo) const;

    private:
	const reco::TrackIPTagInfo::TrackIPData &
	threshTrack(const reco::TrackIPTagInfo &trackIPTagInfo,
	            const reco::TrackIPTagInfo::SortCriteria sort) const;

	double					charmCut;
	reco::TrackIPTagInfo::SortCriteria	sortCriterium;
	TrackSelector				trackPseudoSelector;
	unsigned int				pseudoMultiplicityMin;
	bool					useTrackWeights;
};

#endif // RecoBTag_SecondaryVertex_CombinedSVComputer_h
