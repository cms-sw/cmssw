#ifndef TrackIPData_h
#define TrackIPData_h

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"

namespace reco {


struct TrackIPData
{
  Measurement1D impactParameter3D;
  Measurement1D impactParameter2D;
  //float decayLen;
  //float decayLenError;
};

struct TracksInJetData
{
//const TrackIPData & data(const JetTracksAssociation &jta, edm::Ref<TrackCollection> track) ;
 std::vector<TrackIPData> ipData;
 std::vector<float> probabilities3D; //can we leave empty for HLT usage?
 std::vector<float> probabilities2D; //can we leave empty for HLT usage?
 //edm::Ref<VertexCollection> primaryVertex;
 GlobalVector direction;   //direction used is not forced to be the CaloJet direction
};

typedef edm::AssociationMap<edm::OneToValue<JetTracksAssociationCollection,TracksInJetData> > JetTracksIPDataAssociationCollection;

  typedef
  JetTracksIPDataAssociationCollection::value_type  JetTracksIPDataAssociation;

  typedef
  edm::Ref< JetTracksIPDataAssociationCollection>  JetTracksIPDataAssociationRef;

  typedef
  edm::RefProd<JetTracksIPDataAssociationCollection> JetTracksIPDataAssociationRefProd;

  typedef
  edm::RefVector<JetTracksIPDataAssociationCollection> JetTracksIPDataAssociationRefVector;

}
#endif

