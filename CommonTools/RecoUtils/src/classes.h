#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

namespace { 
  struct dictionary {

    reco::Vertex  v0;
    edm::Wrapper<reco::Vertex> v1;

    reco::Track  t0;
    edm::Wrapper<reco::Track> t1;

    reco::PFCandidate  p0;
    edm::Wrapper<reco::PFCandidate> p1;

    std::vector<reco::Vertex>  vv0;
    edm::Wrapper<std::vector<reco::Vertex> > vv1;

    std::vector<reco::Track>  tv0;
    edm::Wrapper<std::vector<reco::Track> > tv1;

    std::vector<reco::PFCandidate>  pv0;
    edm::Wrapper<std::vector<reco::PFCandidate> > pv1;

    edm::helpers::KeyVal<edm::RefProd<std::vector<reco::Vertex> >,edm::RefProd<std::vector<reco::Track> > > am0;

    edm::helpers::KeyVal<edm::RefProd<std::vector<reco::Vertex> >,edm::RefProd<std::vector<reco::PFCandidate> > > pm0;

    edm::helpers::KeyVal<edm::Ref<std::vector<reco::Vertex>,reco::Vertex,edm::refhelper::FindUsingAdvance<std::vector<reco::Vertex>,reco::Vertex> >,std::vector<std::pair<edm::Ref<std::vector<reco::Track>,reco::Track,edm::refhelper::FindUsingAdvance<std::vector<reco::Track>,reco::Track> >,float> > > am1;

    edm::helpers::KeyVal<edm::Ref<std::vector<reco::Vertex>,reco::Vertex,edm::refhelper::FindUsingAdvance<std::vector<reco::Vertex>,reco::Vertex> >,std::vector<std::pair<edm::Ref<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> >,float> > > pm1;

    edm::AssociationMap<edm::OneToManyWithQuality<std::vector<reco::Vertex>,std::vector<reco::Track>,float,unsigned int> > am2;

    edm::AssociationMap<edm::OneToManyWithQuality<std::vector<reco::Vertex>,std::vector<reco::PFCandidate>,float,unsigned int> > pm2;

    edm::Wrapper<edm::AssociationMap<edm::OneToManyWithQuality<std::vector<reco::Vertex>,std::vector<reco::Track>,float,unsigned int> > > am3;

    edm::Wrapper<edm::AssociationMap<edm::OneToManyWithQuality<std::vector<reco::Vertex>,std::vector<reco::PFCandidate>,float,unsigned int> > > pm3;

    std::map<unsigned int,edm::helpers::KeyVal<edm::Ref<std::vector<reco::Vertex>,reco::Vertex,edm::refhelper::FindUsingAdvance<std::vector<reco::Vertex>,reco::Vertex> >,std::vector<std::pair<edm::Ref<std::vector<reco::Track>,reco::Track,edm::refhelper::FindUsingAdvance<std::vector<reco::Track>,reco::Track> >,float> > > > am4;

    std::map<unsigned int,edm::helpers::KeyVal<edm::Ref<std::vector<reco::Vertex>,reco::Vertex,edm::refhelper::FindUsingAdvance<std::vector<reco::Vertex>,reco::Vertex> >,std::vector<std::pair<edm::Ref<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> >,float> > > > pm4;

    std::vector<std::pair<edm::Ref<std::vector<reco::Track>,reco::Track,edm::refhelper::FindUsingAdvance<std::vector<reco::Track>,reco::Track> >,float> > am5;

    std::vector<std::pair<edm::Ref<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> >,float> > pm5;

    std::pair<edm::Ref<std::vector<reco::Track>,reco::Track,edm::refhelper::FindUsingAdvance<std::vector<reco::Track>,reco::Track> >,float> am6;

    std::pair<edm::Ref<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> >,float> pm6;

    std::vector<std::pair<unsigned int,float> > am7;

    std::map<unsigned int,std::vector<std::pair<unsigned int,float> > > am8;

  };
}

