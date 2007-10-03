#include <cstddef>
#include <vector>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"  
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "DataFormats/BTauReco/interface/VertexTypes.h"

#include "RecoBTag/SecondaryVertex/interface/TrackKinematics.h"

#include "RecoBTag/SecondaryVertex/interface/CombinedSVComputer.h"

using namespace reco;

CombinedSVComputer::CombinedSVComputer(const edm::ParameterSet &params) :
	charmCut(params.getParameter<double>("charmCut"))
{
}

const TrackIPTagInfo::TrackIPData &
CombinedSVComputer::threshTrack(const TrackIPTagInfo &trackIPTagInfo,
                                const TrackIPTagInfo::SortCriteria sort) const
{
	const edm::RefVector<TrackCollection> &tracks =
					trackIPTagInfo.selectedTracks();
	const std::vector<TrackIPTagInfo::TrackIPData> &ipData =
					trackIPTagInfo.impactParameterData();
	std::vector<std::size_t> indices = trackIPTagInfo.sortedIndexes(sort);

	TrackKinematics kin;
	for(std::vector<std::size_t>::const_iterator iter = indices.begin();
	    iter != indices.end(); iter++) {
		kin.add(*tracks[*iter]);

		if (kin.vectorSum().M() > charmCut) 
			return ipData[*iter];
	}

	static TrackIPTagInfo::TrackIPData dummy = {
		  Measurement1D(-999.0, 1.0),
		  Measurement1D(-999.0, 1.0),
	};
	return dummy;
}

TaggingVariableList
CombinedSVComputer::operator () (const TrackIPTagInfo &ipInfo,
                                 const SecondaryVertexTagInfo &svInfo) const
{
	btag::Vertices::VertexType vtxType = btag::Vertices::NoVertex;

	TaggingVariableList variables; // = ipInfo.taggingVariables();

#if 0
	std::vector<std::size_t> indices = ipinfo->sortedIndexes(); // FIXME
	for(std::vector<size_t>::const_iterator iter = indices.begin();
	    iter != indices.end(); ++iter) {
		const TrackIPTagInfo::TrackIPData *data = &m_data[*it];
     vars.insert(TaggingVariable(reco::btau::trackSip3dVal, data->ip3d.value()));
     vars.insert(TaggingVariable(reco::btau::trackSip3dSig, data->ip3d.significance()));
     vars.insert(TaggingVariable(reco::btau::trackSip2dVal, data->ip2d.value()));
     vars.insert(TaggingVariable(reco::btau::trackSip2dSig, data->ip2d.significance()));
//     vars.insert(TaggingVariable(reco::btau::trackDecayLenVal, data->));
//     vars.insert(TaggingVariable(reco::btau::trackDecayLenSig, data->));
     vars.insert(TaggingVariable(reco::btau::trackJetDist, data->distanceToJetAxis));
     vars.insert(TaggingVariable(reco::btau::trackFirstTrackDist, data->distanceToFirstTrack));
   } 
#endif

	if (svInfo.nVertices() > 0) {
		vtxType = btag::Vertices::RecoVertex;
		variables.insert(svInfo.taggingVariables());
	}

	variables.insert(btau::vertexCategory, vtxType, true);
	variables.insert(btau::trackSip3dSigAboveCharm, threshTrack(ipInfo,
			TrackIPTagInfo::IP3DSig).ip3d.significance(), true);
	variables.insert(btau::trackSip2dSigAboveCharm, threshTrack(ipInfo,
			TrackIPTagInfo::IP2DSig).ip2d.significance(), true);

	return variables;
}
