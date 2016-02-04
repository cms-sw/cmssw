#include <functional>
#include <ext/functional>
#include <algorithm>

#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

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

	struct IndexedVertexTrackSelector {
		IndexedVertexTrackSelector(unsigned int index) :
			index(index) {}

		bool operator () (const SecondaryVertexTagInfo::IndexedTrackData &idt)
		{ return idt.second.associatedToVertex(index); }

		unsigned int index;
	};
}

SecondaryVertexTagInfo::SecondaryVertexTagInfo(
                const std::vector<IndexedTrackData> &trackData,
		const std::vector<VertexData> &svData,
		unsigned int vertexCandidates,
		const TrackIPTagInfoRef &trackIPTagInfoRef) :
	m_trackData(trackData),
	m_svData(svData),
	m_vertexCandidates(vertexCandidates),
	m_trackIPTagInfoRef(trackIPTagInfoRef)
{
}

unsigned int SecondaryVertexTagInfo::nVertexTracks() const
{
	return std::count_if(m_trackData.begin(), m_trackData.end(),
	                     VertexTrackSelector());
}

unsigned int SecondaryVertexTagInfo::nVertexTracks(unsigned int index) const
{
	return std::count_if(m_trackData.begin(), m_trackData.end(),
	                     IndexedVertexTrackSelector(index));
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

TrackRefVector SecondaryVertexTagInfo::vertexTracks(unsigned int index) const
{
	TrackRefVector trackRefs;
	const TrackRefVector &trackIPTrackRefs =
				m_trackIPTagInfoRef->selectedTracks();

	for(std::vector<IndexedTrackData>::const_iterator iter =
		m_trackData.begin(); iter != m_trackData.end(); iter++)

		if (iter->second.associatedToVertex(index))
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

float SecondaryVertexTagInfo::trackWeight(unsigned int svIndex,
                                          const TrackRef &track) const
{
	return m_svData[svIndex].vertex.trackWeight(track);
}

float SecondaryVertexTagInfo::trackWeight(unsigned int svIndex,
                                          unsigned int trackIndex) const
{
	return trackWeight(svIndex, track(trackIndex));
}

TaggingVariableList SecondaryVertexTagInfo::taggingVariables() const
{
	TaggingVariableList vars;

	for(std::vector<VertexData>::const_iterator iter = m_svData.begin();
	    iter != m_svData.end(); iter++) {
		vars.insert(btau::flightDistance2dVal,
					iter->dist2d.value(), true);
		vars.insert(btau::flightDistance2dSig,
					iter->dist2d.significance(), true);
		vars.insert(btau::flightDistance3dVal,
					iter->dist3d.value(), true);
		vars.insert(btau::flightDistance3dSig,
					iter->dist3d.significance(), true);

		vars.insert(btau::vertexJetDeltaR,
			Geom::deltaR(iter->direction, jet()->momentum()), true);
	}

	vars.insert(btau::jetNSecondaryVertices, m_vertexCandidates, true);
	vars.insert(btau::vertexNTracks, nVertexTracks(), true);

	vars.finalize();
	return vars;
}
