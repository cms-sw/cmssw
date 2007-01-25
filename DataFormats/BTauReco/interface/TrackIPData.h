#ifndef TrackIPData_h
#define TrackIPData_h

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Common/interface/Ref.h"
#include "Geometry/Vector/interface/GlobalVector.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

namespace reco {


struct TrackIPData
{
  float impactParameter3D;
  float impactParameter3DError;
  float impactParameter2D;
  float impactParameter2DError;
  //float decayLen;
  //float decayLenError;
};

struct TracksInJetData
{
 std::vector<TrackIPData> ipData;
 std::vector<float> probabilities3D; //can we leave empty for HLT usage?
 std::vector<float> probabilities2D; //can we leave empty for HLT usage?
 edm::Ref<VertexCollection> primaryVertex;
 GlobalVector direction;   //direction used is not forced to be the CaloJet direction
};

typedef edm::AssociationMap<edm::OneToValue<JetTracksAssociationCollection,TracksInJetData> > JetTracksIPDataAssociationCollection;

}
#endif

