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
 
class SecondaryVertexTagInfo : public BaseTagInfo {
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

	SecondaryVertexTagInfo() {}
	virtual ~SecondaryVertexTagInfo() {}

	SecondaryVertexTagInfo(
	                const std::vector<IndexedTrackData> &trackData,
			const std::vector<VertexData> &svData,
			unsigned int vertexCandidates,
			const TrackIPTagInfoRef &trackIPTagInfoRef);

        /// clone
        virtual SecondaryVertexTagInfo * clone(void) const {
            return new SecondaryVertexTagInfo(*this);
        }
  
	const TrackIPTagInfoRef &trackIPTagInfoRef() const
	{ return m_trackIPTagInfoRef; }

	virtual edm::RefToBase<Jet> jet(void) const
	{ return m_trackIPTagInfoRef->jet(); }

	virtual TrackRefVector tracks(void) const
	{ return m_trackIPTagInfoRef->tracks(); }

	const JetTracksAssociationRef &jtaRef(void) const
	{ return m_trackIPTagInfoRef->jtaRef(); }

	const Vertex &secondaryVertex(unsigned int index) const
	{ return m_svData[index].vertex; }

	unsigned int nSelectedTracks() const { return m_trackData.size(); }
	unsigned int nVertexTracks() const;
	unsigned int nVertexTracks(unsigned int index) const;
	unsigned int nVertices() const { return m_svData.size(); }
	unsigned int nVertexCandidates() const { return m_vertexCandidates; }

	TrackRefVector selectedTracks() const;
	TrackRefVector vertexTracks() const;
	TrackRefVector vertexTracks(unsigned int index) const;

	TrackRef track(unsigned int index) const;
	unsigned int findTrack(const TrackRef &track) const;

	const TrackData &trackData(unsigned int index) const;
	const TrackData &trackData(const TrackRef &track) const;

	const TrackIPTagInfo::TrackIPData &trackIPData(unsigned int index) const;
	const TrackIPTagInfo::TrackIPData &trackIPData(const TrackRef &track) const;

	float trackWeight(unsigned int svIndex, unsigned int trackindex) const;
	float trackWeight(unsigned int svIndex, const TrackRef &track) const;

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

	TrackIPTagInfoRef			m_trackIPTagInfoRef;
};

DECLARE_EDM_REFS(SecondaryVertexTagInfo)

}

#endif // DataFormats_BTauReco_SecondaryVertexTagInfo_h
