#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/NuclearInteraction.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <vector>
#include <utility>

namespace {
  struct dictionary {
    reco::Vertex rv1;
    std::vector<reco::Vertex> v1;
    edm::Wrapper<std::vector<reco::Vertex> > wc1;
    edm::Ref<std::vector<reco::Vertex> > r1;
    edm::RefProd<std::vector<reco::Vertex> > rp1;
    edm::RefVector<std::vector<reco::Vertex> > rvv1;

    reco::NuclearInteraction nrv1;
    std::vector<reco::NuclearInteraction> nv1;
    edm::Wrapper<std::vector<reco::NuclearInteraction> > nwc1;
    edm::Ref<std::vector<reco::NuclearInteraction> > nr1;
    edm::RefProd<std::vector<reco::NuclearInteraction> > nrp1;
    edm::RefVector<std::vector<reco::NuclearInteraction> > nrvv1;

    edm::helpers::KeyVal<edm::RefProd<std::vector<reco::Vertex> >,edm::RefProd<std::vector<reco::Track> > > am0;
    edm::helpers::KeyVal<edm::Ref<std::vector<reco::Vertex>,reco::Vertex,edm::refhelper::FindUsingAdvance<std::vector<reco::Vertex>,reco::Vertex> >,std::vector<std::pair<edm::Ref<std::vector<reco::Track>,reco::Track,edm::refhelper::FindUsingAdvance<std::vector<reco::Track>,reco::Track> >,float> > > amf1;
    edm::AssociationMap<edm::OneToManyWithQuality<std::vector<reco::Vertex>,std::vector<reco::Track>,float,unsigned int> > amf2;
    edm::Wrapper<edm::AssociationMap<edm::OneToManyWithQuality<std::vector<reco::Vertex>,std::vector<reco::Track>,float,unsigned int> > > amf3;
    std::map<unsigned int,edm::helpers::KeyVal<edm::Ref<std::vector<reco::Vertex>,reco::Vertex,edm::refhelper::FindUsingAdvance<std::vector<reco::Vertex>,reco::Vertex> >,std::vector<std::pair<edm::Ref<std::vector<reco::Track>,reco::Track,edm::refhelper::FindUsingAdvance<std::vector<reco::Track>,reco::Track> >,float> > > > amf4;
    edm::helpers::KeyVal<edm::Ref<std::vector<reco::Vertex>,reco::Vertex,edm::refhelper::FindUsingAdvance<std::vector<reco::Vertex>,reco::Vertex> >,std::vector<std::pair<edm::Ref<std::vector<reco::Track>,reco::Track,edm::refhelper::FindUsingAdvance<std::vector<reco::Track>,reco::Track> >,int> > > am1;
    edm::AssociationMap<edm::OneToManyWithQuality<std::vector<reco::Vertex>,std::vector<reco::Track>,int,unsigned int> > am2;
    edm::Wrapper<edm::AssociationMap<edm::OneToManyWithQuality<std::vector<reco::Vertex>,std::vector<reco::Track>,int,unsigned int> > > am3;
    std::map<unsigned int,edm::helpers::KeyVal<edm::Ref<std::vector<reco::Vertex>,reco::Vertex,edm::refhelper::FindUsingAdvance<std::vector<reco::Vertex>,reco::Vertex> >,std::vector<std::pair<edm::Ref<std::vector<reco::Track>,reco::Track,edm::refhelper::FindUsingAdvance<std::vector<reco::Track>,reco::Track> >,int> > > > am4;

    std::vector<std::pair<edm::Ref<std::vector<reco::Vertex>,reco::Vertex,edm::refhelper::FindUsingAdvance<std::vector<reco::Vertex>,reco::Vertex> >,int> > ma5;
    std::pair<edm::Ref<std::vector<reco::Vertex>,reco::Vertex,edm::refhelper::FindUsingAdvance<std::vector<reco::Vertex>,reco::Vertex> >,int> ma6;

  };
}
