#ifndef RecoBTag_SecondaryVertex_SimpleSecondaryVertexComputer_h
#define RecoBTag_SecondaryVertex_SimpleSecondaryVertexComputer_h

#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoBTag/SecondaryVertex/interface/TemplatedSimpleSecondaryVertexComputer.h"


typedef TemplatedSimpleSecondaryVertexComputer<reco::TrackIPTagInfo,reco::Vertex> SimpleSecondaryVertexComputer;

#endif // RecoBTag_SecondaryVertex_SimpleSecondaryVertexComputer_h
