#ifndef DataFormats_BTauReco_SecondaryVertexTagInfo_h
#define DataFormats_BTauReco_SecondaryVertexTagInfo_h

#include <utility>
#include <vector>

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"

namespace reco {
template <class IPTI> 
class TemplatedSecondaryVertexTagInfo : public BaseTagInfo {
    public:
	struct TrackData {
		enum Status {
			trackSelected = 0,
			trackUsedForVertexFit,
			trackAssociatedToVertex
		};

		inline bool usedForVertexFit() const
		{ return (int)svStatus >= (int)trackUsedForVertexFit; }
		inline bool associatedToVertex() const
		{ return (int)svStatus >= (int)trackAssociatedToVertex; }
		inline bool associatedToVertex(unsigned int index) const
		{ return (int)svStatus == (int)trackAssociatedToVertex + (int)index; }

		Status	svStatus;
	};

	struct VertexData {
		Vertex				vertex;
		Measurement1D			dist2d, dist3d;
		GlobalVector			direction;
	};

	typedef std::pair<unsigned int, TrackData> IndexedTrackData;
        struct TrackFinder {
                TrackFinder(const TrackRefVector &tracks,
                            const TrackRef &track) :
                        tracks(tracks), track(track) {}

                bool operator () (const IndexedTrackData &idt)
                { return tracks[idt.first] == track; }

                const TrackRefVector    &tracks;
                const TrackRef          &track;
        };

        struct VertexTrackSelector {
                bool operator () (const IndexedTrackData &idt)
                { return idt.second.associatedToVertex(); }
        };

        struct IndexedVertexTrackSelector {
                IndexedVertexTrackSelector(unsigned int index) :
                        index(index) {}

                bool operator () (const IndexedTrackData &idt)
                { return idt.second.associatedToVertex(index); }

                unsigned int index;
        };



	typedef typename IPTI::input_container input_container;

	TemplatedSecondaryVertexTagInfo() {}
	virtual ~TemplatedSecondaryVertexTagInfo() {}

	TemplatedSecondaryVertexTagInfo(
	                const std::vector<IndexedTrackData> &trackData,
			const std::vector<VertexData> &svData,
			unsigned int vertexCandidates,
			const IPTI &trackIPTagInfoRef);

        /// clone
        virtual TemplatedSecondaryVertexTagInfo * clone(void) const {
            return new TemplatedSecondaryVertexTagInfo(*this);
        }
  
	const TrackIPTagInfoRef &trackIPTagInfoRef() const
	{ return m_trackIPTagInfoRef; }

	virtual edm::RefToBase<Jet> jet(void) const
	{ return m_trackIPTagInfoRef->jet(); }

	virtual input_container tracks(void) const
	{ return m_trackIPTagInfoRef->tracks(); }

//AR TODO is it needed?
//	const JetTracksAssociationRef &jtaRef(void) const
//	{ return m_trackIPTagInfoRef->jtaRef(); }

	const Vertex &secondaryVertex(unsigned int index) const
	{ return m_svData[index].vertex; }

	unsigned int nSelectedTracks() const { return m_trackData.size(); }
	unsigned int nVertexTracks() const;
	unsigned int nVertexTracks(unsigned int index) const;
	unsigned int nVertices() const { return m_svData.size(); }
	unsigned int nVertexCandidates() const { return m_vertexCandidates; }

	input_container selectedTracks() const;
	input_container vertexTracks() const;
	input_container vertexTracks(unsigned int index) const;

	typename input_container::value_type track(unsigned int index) const;
	unsigned int findTrack(const typename input_container::value_type &track) const;

	const TrackData &trackData(unsigned int index) const;
	const TrackData &trackData(const typename input_container::value_type &track) const;

	const typename IPTI::TrackIPData &trackIPData(unsigned int index) const;
	const typename IPTI::TrackIPData &trackIPData(const typename input_container::value_type &track) const;

	float trackWeight(unsigned int svIndex, unsigned int trackindex) const;
	float trackWeight(unsigned int svIndex, const typename input_container::value_type &track) const;

	Measurement1D
	flightDistance(unsigned int index, bool in2d = false) const
	{ return in2d ? m_svData[index].dist2d : m_svData[index].dist3d; }
	const GlobalVector &flightDirection(unsigned int index) const
	{ return m_svData[index].direction; }

	virtual TaggingVariableList taggingVariables() const;

    private:
	std::vector<IndexedTrackData>		m_trackData;
	std::vector<VertexData>			m_svData;
	unsigned int				m_vertexCandidates;

	edm::Ref<std::vector<IPTI> >		m_trackIPTagInfoRef;
};

typedef TemplatedSecondaryVertexTagInfo<TrackIPTagInfo> SecondaryVertexTagInfo;
//typedef TemplatedSecondaryVertexTagInfo<CandIPTagInfo> CandSecondaryVertexTagInfo;

DECLARE_EDM_REFS(SecondaryVertexTagInfo)

}

#include <functional>
#include <ext/functional>
#include <algorithm>

#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"

using namespace reco;


template<class IPTI>  TemplatedSecondaryVertexTagInfo<IPTI>::TemplatedSecondaryVertexTagInfo(
                const std::vector<IndexedTrackData> &trackData,
		const std::vector<VertexData> &svData,
		unsigned int vertexCandidates,
		const IPTI &trackIPTagInfoRef) :
	m_trackData(trackData),
	m_svData(svData),
	m_vertexCandidates(vertexCandidates),
	m_trackIPTagInfoRef(trackIPTagInfoRef)
{
}

template<class IPTI> unsigned int  TemplatedSecondaryVertexTagInfo<IPTI>::nVertexTracks() const
{
	return std::count_if(m_trackData.begin(), m_trackData.end(),
	                     VertexTrackSelector());
}

template<class IPTI> unsigned int  TemplatedSecondaryVertexTagInfo<IPTI>::nVertexTracks(unsigned int index) const
{
	return std::count_if(m_trackData.begin(), m_trackData.end(),
	                     IndexedVertexTrackSelector(index));
}

template<class IPTI> 
typename reco::TemplatedSecondaryVertexTagInfo<IPTI>::input_container  TemplatedSecondaryVertexTagInfo<IPTI>::selectedTracks() const
{
	input_container trackRefs;
	const input_container &trackIPTrackRefs =
				m_trackIPTagInfoRef->selectedTracks();

	for(typename std::vector<typename reco::TemplatedSecondaryVertexTagInfo<IPTI>::IndexedTrackData>::const_iterator iter =
		m_trackData.begin(); iter != m_trackData.end(); iter++)

		trackRefs.push_back(trackIPTrackRefs[iter->first]);

	return trackRefs;
}

template<class IPTI> 
typename TemplatedSecondaryVertexTagInfo<IPTI>::input_container  TemplatedSecondaryVertexTagInfo<IPTI>::vertexTracks() const
{
	input_container trackRefs;
	const input_container &trackIPTrackRefs =
				m_trackIPTagInfoRef->selectedTracks();

	for(typename std::vector<typename reco::TemplatedSecondaryVertexTagInfo<IPTI>::IndexedTrackData>::const_iterator iter =
		m_trackData.begin(); iter != m_trackData.end(); iter++)

		if (iter->second.associatedToVertex())
			trackRefs.push_back(trackIPTrackRefs[iter->first]);

	return trackRefs;
}  

template<class IPTI> 
typename TemplatedSecondaryVertexTagInfo<IPTI>::input_container  TemplatedSecondaryVertexTagInfo<IPTI>::vertexTracks(unsigned int index) const
{
	input_container trackRefs;
	const input_container &trackIPTrackRefs =
				m_trackIPTagInfoRef->selectedTracks();

	for(typename std::vector<typename reco::TemplatedSecondaryVertexTagInfo<IPTI>::IndexedTrackData>::const_iterator iter =
		m_trackData.begin(); iter != m_trackData.end(); iter++)

		if (iter->second.associatedToVertex(index))
			trackRefs.push_back(trackIPTrackRefs[iter->first]);

	return trackRefs;
}  

template<class IPTI> typename TemplatedSecondaryVertexTagInfo<IPTI>::input_container::value_type  TemplatedSecondaryVertexTagInfo<IPTI>::track(unsigned int index) const
{
	return m_trackIPTagInfoRef->selectedTracks()[m_trackData[index].first];
}

template<class IPTI> unsigned int  TemplatedSecondaryVertexTagInfo<IPTI>::findTrack(const typename input_container::value_type &track) const
{
	typename std::vector<typename reco::TemplatedSecondaryVertexTagInfo<IPTI>::IndexedTrackData>::const_iterator pos =
		std::find_if(m_trackData.begin(), m_trackData.end(),
		             TrackFinder(m_trackIPTagInfoRef->selectedTracks(),
		                         track));

	if (pos == m_trackData.end())
		throw edm::Exception(edm::errors::InvalidReference)
			<< "Track not found in "
			   " TemplatedSecondaryVertexTagInfo<IPTI>::findTrack." << std::endl;

	return pos - m_trackData.begin();
}

template<class IPTI>
const typename   TemplatedSecondaryVertexTagInfo<IPTI>::TrackData&  TemplatedSecondaryVertexTagInfo<IPTI>::trackData(unsigned int index) const
{
	return m_trackData[index].second;
}

template<class IPTI>
const typename  TemplatedSecondaryVertexTagInfo<IPTI>::TrackData&  TemplatedSecondaryVertexTagInfo<IPTI>::trackData(const typename input_container::value_type &track) const
{
	return m_trackData[findTrack(track)].second;
}

template<class IPTI> 
const typename IPTI::TrackIPData& TemplatedSecondaryVertexTagInfo<IPTI>::trackIPData(unsigned int index) const
{
	return m_trackIPTagInfoRef->impactParameterData()[
						m_trackData[index].first];
}

template<class IPTI>
const typename IPTI::TrackIPData& TemplatedSecondaryVertexTagInfo<IPTI>::trackIPData(const typename input_container::value_type &track) const
{
	return trackIPData(findTrack(track));
}

template<class IPTI> float  TemplatedSecondaryVertexTagInfo<IPTI>::trackWeight(unsigned int svIndex,
                                          const typename input_container::value_type &track) const
{
	return m_svData[svIndex].vertex.trackWeight(track);
}

template<class IPTI> float  TemplatedSecondaryVertexTagInfo<IPTI>::trackWeight(unsigned int svIndex,
                                          unsigned int trackIndex) const
{
	return trackWeight(svIndex, track(trackIndex));
}

template<class IPTI> TaggingVariableList  TemplatedSecondaryVertexTagInfo<IPTI>::taggingVariables() const
{
	TaggingVariableList vars;

	for(typename std::vector<typename TemplatedSecondaryVertexTagInfo<IPTI>::VertexData>::const_iterator iter = m_svData.begin();
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
#endif // DataFormats_BTauReco_SecondaryVertexTagInfo_h
