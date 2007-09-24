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

		Status	svStatus;
	};

	typedef std::pair<unsigned int, TrackData> IndexedTrackData;

	SecondaryVertexTagInfo() {}
	virtual ~SecondaryVertexTagInfo() {}

	SecondaryVertexTagInfo(
	                const std::vector<IndexedTrackData> &trackData,
			const Vertex &secondaryVertex,
			Measurement1D dist2d, Measurement1D dist3d,
			unsigned int vertexCount, GlobalVector direction,
			const TrackIPTagInfoRef &trackIPTagInfoRef);

	const TrackIPTagInfoRef &trackIPTagInfoRef() const
	{ return m_trackIPTagInfoRef; }

	virtual edm::RefToBase<Jet> jet(void) const
	{ return m_trackIPTagInfoRef->jet(); }

	virtual TrackRefVector tracks(void) const
	{ return m_trackIPTagInfoRef->tracks(); }

	const JetTracksAssociationRef &jtaRef(void) const
	{ return m_trackIPTagInfoRef->jtaRef(); }

	const Vertex &secondaryVertex() const
	{ return m_secondaryVertex; }

	unsigned int nSelectedTracks() const { return m_trackData.size(); }
	unsigned int nVertexTracks() const;
	unsigned int nVertices() const { return m_vertexCount; }

	TrackRefVector selectedTracks() const;
	TrackRefVector vertexTracks() const;

	TrackRef track(unsigned int index) const;
	unsigned int findTrack(const TrackRef &track) const;

	const TrackData &trackData(unsigned int index) const;
	const TrackData &trackData(const TrackRef &track) const;

	const TrackIPTagInfo::TrackIPData &trackIPData(unsigned int index) const;
	const TrackIPTagInfo::TrackIPData &trackIPData(const TrackRef &track) const;

	float trackWeight(unsigned int index) const;
	float trackWeight(const TrackRef &track) const;

	Measurement1D flightDistance(bool in3d = true) const
	{ return in3d ? m_dist3d : m_dist2d; }
	const GlobalVector &flightDirection() const
	{ return m_direction;}

	virtual TaggingVariableList taggingVariables() const;

    private:
	std::vector<IndexedTrackData>		m_trackData;
	Vertex					m_secondaryVertex;
	unsigned int				m_vertexCount;

	Measurement1D				m_dist2d, m_dist3d;
	GlobalVector				m_direction;

	TrackIPTagInfoRef			m_trackIPTagInfoRef;
};

DECLARE_EDM_REFS(SecondaryVertexTagInfo)

}

#endif // DataFormats_BTauReco_SecondaryVertexTagInfo_h
