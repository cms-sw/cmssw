#include <functional>
#include <ext/functional>
#include <algorithm>
#include <utility>
#include <vector>

#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"

#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"

using namespace reco;

namespace {
	struct TrackFinder {
		TrackFinder(const TrackRefVector &tracks,
		            const TrackRef &track) :
			tracks(tracks), track(track) {}

		bool operator () (const SecondaryVertexTagInfo::IndexedTrackData &idt)
		{ return tracks[idt.first] == track; }

		const TrackRefVector	&tracks;
		const TrackRef		&track;
	};

	struct VertexTrackSelector {
		bool operator () (const SecondaryVertexTagInfo::IndexedTrackData &idt)
		{ return idt.second.associatedToVertex(); }
	};
}

SecondaryVertexTagInfo::SecondaryVertexTagInfo(
                const std::vector<IndexedTrackData> &trackData,
		const Vertex &secondaryVertex,
		Measurement1D dist2d, Measurement1D dist3d,
		unsigned int vertexCount, GlobalVector direction,
		const TrackIPTagInfoRef &trackIPTagInfoRef) :
	m_trackData(trackData),
	m_secondaryVertex(secondaryVertex),
	m_vertexCount(vertexCount),
	m_dist2d(dist2d), m_dist3d(dist3d),
	m_direction(direction),
	m_trackIPTagInfoRef(trackIPTagInfoRef)
{
}

unsigned int SecondaryVertexTagInfo::nVertexTracks() const
{
	return std::count_if(m_trackData.begin(), m_trackData.end(),
	                     VertexTrackSelector());
}

TrackRefVector SecondaryVertexTagInfo::selectedTracks() const
{
	TrackRefVector trackRefs;
	const TrackRefVector &trackIPTrackRefs =
				m_trackIPTagInfoRef->selectedTracks();

	for(std::vector<IndexedTrackData>::const_iterator iter =
		m_trackData.begin(); iter != m_trackData.end(); iter++)

		trackRefs.push_back(trackIPTrackRefs[iter->first]);

	return trackRefs;
}

TrackRefVector SecondaryVertexTagInfo::vertexTracks() const
{
	TrackRefVector trackRefs;
	const TrackRefVector &trackIPTrackRefs =
				m_trackIPTagInfoRef->selectedTracks();

	for(std::vector<IndexedTrackData>::const_iterator iter =
		m_trackData.begin(); iter != m_trackData.end(); iter++)

		if (iter->second.associatedToVertex())
			trackRefs.push_back(trackIPTrackRefs[iter->first]);

	return trackRefs;
}  

TrackRef SecondaryVertexTagInfo::track(unsigned int index) const
{
	return m_trackIPTagInfoRef->selectedTracks()[m_trackData[index].first];
}

unsigned int SecondaryVertexTagInfo::findTrack(const TrackRef &track) const
{
	std::vector<IndexedTrackData>::const_iterator pos =
		std::find_if(m_trackData.begin(), m_trackData.end(),
		             TrackFinder(m_trackIPTagInfoRef->selectedTracks(),
		                         track));

	if (pos == m_trackData.end())
		throw edm::Exception(edm::errors::InvalidReference)
			<< "Track not found in "
			   "SecondaryVertexTagInfo::findTrack." << std::endl;

	return pos - m_trackData.begin();
}

const SecondaryVertexTagInfo::TrackData&
SecondaryVertexTagInfo::trackData(unsigned int index) const
{
	return m_trackData[index].second;
}

const SecondaryVertexTagInfo::TrackData&
SecondaryVertexTagInfo::trackData(const TrackRef &track) const
{
	return m_trackData[findTrack(track)].second;
}

const TrackIPTagInfo::TrackIPData&
SecondaryVertexTagInfo::trackIPData(unsigned int index) const
{
	return m_trackIPTagInfoRef->impactParameterData()[
						m_trackData[index].first];
}

const TrackIPTagInfo::TrackIPData&
SecondaryVertexTagInfo::trackIPData(const TrackRef &track) const
{
	return trackIPData(findTrack(track));
}

float SecondaryVertexTagInfo::trackWeight(const TrackRef &track) const
{
	return m_secondaryVertex.trackWeight(track);
}

float SecondaryVertexTagInfo::trackWeight(unsigned int index) const
{
	return trackWeight(track(index));
}

TaggingVariableList SecondaryVertexTagInfo::taggingVariables() const
{
	TaggingVariableList vars;

	return vars;
}
