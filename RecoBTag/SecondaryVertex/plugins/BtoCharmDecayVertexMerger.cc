// this code is partially taken from BCandidateProducer of L. Wehrli

// first revised prototype version for CMSSW 29.Aug.2012

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/BTauReco/interface/ParticleMasses.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoBTag/SecondaryVertex/interface/SecondaryVertex.h"

#include "DataFormats/GeometryVector/interface/VectorUtil.h"

#include <algorithm>

reco::Candidate::LorentzVector vtxP4(const reco::VertexCompositePtrCandidate & vtx) {
  reco::Candidate::LorentzVector sum;
  const std::vector<reco::CandidatePtr> & tracks = vtx.daughterPtrVector();

  for(std::vector<reco::CandidatePtr>::const_iterator track = tracks.begin(); track != tracks.end(); ++track) {
    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > vec;
    vec.SetPx((*track)->px());
    vec.SetPy((*track)->py());
    vec.SetPz((*track)->pz());
    vec.SetM(reco::ParticleMasses::piPlus);
    sum += vec;
  }
  return sum;
}

template<typename VTX>
class BtoCharmDecayVertexMergerT : public edm::stream::EDProducer<> {
    public:
       BtoCharmDecayVertexMergerT(const edm::ParameterSet &params);

       virtual void produce(edm::Event &event, const edm::EventSetup &es);

       typedef reco::TemplatedSecondaryVertex<VTX> SecondaryVertex;

    private:
       edm::EDGetTokenT<reco::VertexCollection>  token_primaryVertex;
       edm::EDGetTokenT<edm::View<VTX> >         token_secondaryVertex;
       reco::Vertex                              pv;
       // double                                    maxFraction;
       // double                                    minSignificance;

       double maxDRForUnique, maxvecSumIMCUTForUnique, minCosPAtomerge, maxPtreltomerge;

       // a vertex proxy for sorting using stl
       struct VertexProxy{
         VTX vert;
         double invm;
         size_t ntracks;
       };

       // comparison operator for VertexProxy, used in sorting
       friend bool operator<(VertexProxy v1, VertexProxy v2){
         if(v1.ntracks>2 && v2.ntracks<3) return true;
         if(v1.ntracks<3 && v2.ntracks>2) return false;
         return (v1.invm>v2.invm);
       }

       VertexProxy buildVertexProxy(const VTX & vtx);

       void resolveBtoDchain(std::vector<VertexProxy>& coll, unsigned int i, unsigned int k);
       GlobalVector flightDirection(const reco::Vertex &v1, const reco::Vertex &v2);
       GlobalVector flightDirection(const reco::Vertex &v1, const reco::VertexCompositePtrCandidate &v2);
       GlobalVector flightDirection(const reco::VertexCompositePtrCandidate &v1, const reco::VertexCompositePtrCandidate &v2);
};


template<typename VTX>
BtoCharmDecayVertexMergerT<VTX>::BtoCharmDecayVertexMergerT(const edm::ParameterSet &params) :
       token_primaryVertex(consumes<reco::VertexCollection>(params.getParameter<edm::InputTag>("primaryVertices"))),
       token_secondaryVertex(consumes<edm::View<VTX> >(params.getParameter<edm::InputTag>("secondaryVertices"))),
       maxDRForUnique(params.getParameter<double>("maxDRUnique")),
       maxvecSumIMCUTForUnique(params.getParameter<double>("maxvecSumIMifsmallDRUnique")),
       minCosPAtomerge(params.getParameter<double>("minCosPAtomerge")),
       maxPtreltomerge(params.getParameter<double>("maxPtreltomerge"))
       // maxFraction(params.getParameter<double>("maxFraction")),
       // minSignificance(params.getParameter<double>("minSignificance"))
{
       produces<std::vector<VTX> >();
}
//------------------------------------
template<typename VTX>
void BtoCharmDecayVertexMergerT<VTX>::produce(edm::Event &iEvent, const edm::EventSetup &iSetup){

  using namespace reco;
  // PV
  edm::Handle<reco::VertexCollection> PVcoll;
  iEvent.getByToken(token_primaryVertex, PVcoll);

  if(PVcoll->size()!=0) {

  const reco::VertexCollection pvc = *( PVcoll.product());
  pv = pvc[0];

  // get the IVF collection
  edm::Handle<edm::View<VTX> > secondaryVertices;
  iEvent.getByToken(token_secondaryVertex, secondaryVertices);



  //loop over vertices,  fill into own collection for sorting
  std::vector<VertexProxy> vertexProxyColl;
  for(typename edm::View<VTX>::const_iterator sv = secondaryVertices->begin();
      sv != secondaryVertices->end(); ++sv) {
    vertexProxyColl.push_back( buildVertexProxy(*sv) );
  }

  // sort the vertices by mass and track multiplicity
  sort( vertexProxyColl.begin(), vertexProxyColl.end());


  // loop forward over all vertices
  for(unsigned int iVtx=0; iVtx < vertexProxyColl.size(); ++iVtx){

    // nested loop backwards (in order to start from light masses)
    // check all vertices against each other for B->D chain
    for(unsigned int kVtx=vertexProxyColl.size()-1; kVtx>iVtx; --kVtx){
      // remove D vertices from the collection and add the tracks to the original one
      resolveBtoDchain(vertexProxyColl, iVtx, kVtx);
    }
  }

  // now create new vertex collection and add to event
  std::auto_ptr<std::vector<VTX> > bvertColl(new std::vector<VTX>);
  for(typename std::vector<VertexProxy>::const_iterator it=vertexProxyColl.begin(); it!=vertexProxyColl.end(); ++it) bvertColl->push_back((*it).vert);
  iEvent.put(bvertColl);
  }
  else{
    std::auto_ptr<std::vector<VTX> > bvertCollEmpty(new std::vector<VTX>);
    iEvent.put(bvertCollEmpty);
  }
}
//------------------------------------
template<typename VTX>
GlobalVector
BtoCharmDecayVertexMergerT<VTX>::flightDirection(const reco::Vertex & v1, const reco::Vertex & v2) {
  return  GlobalVector(v2.x() - v1.x(),
                       v2.y() - v1.y(),
                       v2.z() - v1.z());
}

template<typename VTX>
GlobalVector
BtoCharmDecayVertexMergerT<VTX>::flightDirection(const reco::Vertex & v1, const reco::VertexCompositePtrCandidate & v2) {
  return  GlobalVector(v2.vertex().x() - v1.x(),
                       v2.vertex().y() - v1.y(),
                       v2.vertex().z() - v1.z());
}

template<typename VTX>
GlobalVector
BtoCharmDecayVertexMergerT<VTX>::flightDirection(const reco::VertexCompositePtrCandidate & v1, const reco::VertexCompositePtrCandidate & v2) {
  return  GlobalVector(v2.vertex().x() - v1.vertex().x(),
                       v2.vertex().y() - v1.vertex().y(),
                       v2.vertex().z() - v1.vertex().z());
}
//-------------- template specializations --------------------

// -------------- buildVertexProxy ----------------
template<>
BtoCharmDecayVertexMergerT<reco::Vertex>::VertexProxy
BtoCharmDecayVertexMergerT<reco::Vertex>::buildVertexProxy(const reco::Vertex & vtx) {
  VertexProxy vtxProxy = {vtx,vtx.p4().M(),vtx.tracksSize()};
  return vtxProxy;
}

template<>
BtoCharmDecayVertexMergerT<reco::VertexCompositePtrCandidate>::VertexProxy
BtoCharmDecayVertexMergerT<reco::VertexCompositePtrCandidate>::buildVertexProxy(const reco::VertexCompositePtrCandidate & vtx) {
  VertexProxy vtxProxy = {vtx,vtxP4(vtx).M(),vtx.numberOfSourceCandidatePtrs()};
  return vtxProxy;
}

// -------------- resolveBtoDchain ----------------
template<>
void BtoCharmDecayVertexMergerT<reco::Vertex>::resolveBtoDchain(std::vector<VertexProxy>& coll, unsigned int i, unsigned int k){
  using namespace reco;

  GlobalVector momentum1 = GlobalVector(coll[i].vert.p4().X(), coll[i].vert.p4().Y(), coll[i].vert.p4().Z());
  GlobalVector momentum2 = GlobalVector(coll[k].vert.p4().X(), coll[k].vert.p4().Y(), coll[k].vert.p4().Z());

  SecondaryVertex sv1(pv, coll[i].vert, momentum1 , true);
  SecondaryVertex sv2(pv, coll[k].vert, momentum2 , true);


  // find out which one is near and far
  SecondaryVertex svNear = sv1;
  SecondaryVertex svFar  = sv2;
  GlobalVector momentumNear  = momentum1;
  GlobalVector momentumFar   = momentum2;

  // swap if it is the other way around
  if(sv1.dist3d().value() >= sv2.dist3d().value()){
    svNear = sv2;
    svFar = sv1;
    momentumNear = momentum2;
    momentumFar = momentum1;
  }
  GlobalVector nearToFar = flightDirection( svNear, svFar);
  GlobalVector pvToNear  = flightDirection( pv, svNear);
  GlobalVector pvToFar   = flightDirection( pv, svFar);

  double cosPA =  nearToFar.dot(momentumFar) / momentumFar.mag()/ nearToFar.mag();
  double cosa  =  pvToNear. dot(momentumFar) / pvToNear.mag()   / momentumFar.mag();
  double ptrel = sqrt(1.0 - cosa*cosa)* momentumFar.mag();

  double vertexDeltaR = Geom::deltaR(pvToNear, pvToFar);

  // create a set of all tracks from both vertices, avoid double counting by using a std::set<>
  std::set<reco::TrackRef> trackrefs;
  // first vertex
  for(reco::Vertex::trackRef_iterator ti = sv1.tracks_begin(); ti!=sv1.tracks_end(); ++ti){
    if(sv1.trackWeight(*ti)>0.5){
      reco::TrackRef t = ti->castTo<reco::TrackRef>();
      trackrefs.insert(t);
    }
  }
  // second vertex
  for(reco::Vertex::trackRef_iterator ti = sv2.tracks_begin(); ti!=sv2.tracks_end(); ++ti){
    if(sv2.trackWeight(*ti)>0.5){
      reco::TrackRef t = ti->castTo<reco::TrackRef>();
      trackrefs.insert(t);
    }
  }

  // now calculate one LorentzVector from the track momenta
  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > mother;
  for(std::set<reco::TrackRef>::const_iterator it = trackrefs.begin(); it!= trackrefs.end(); ++it){
    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> >  temp ( (*it)->px(),(*it)->py(),(*it)->pz(), ParticleMasses::piPlus );
    mother += temp;
  }


  //   check if criteria for merging are fulfilled
  if(vertexDeltaR<maxDRForUnique && mother.M()<maxvecSumIMCUTForUnique && cosPA>minCosPAtomerge && std::abs(ptrel)<maxPtreltomerge ) {


    // add tracks of second vertex which are missing to first vertex
    // loop over the second
    bool bFoundDuplicate=false;
    for(reco::Vertex::trackRef_iterator ti = sv2.tracks_begin(); ti!=sv2.tracks_end(); ++ti){
      reco::Vertex::trackRef_iterator it = find(sv1.tracks_begin(), sv1.tracks_end(), *ti);
      if (it==sv1.tracks_end()) coll[i].vert.add( *ti, sv2.refittedTrack(*ti), sv2.trackWeight(*ti) );
      else bFoundDuplicate=true;
    }
    // in case a duplicate track is found, need to create the full track list in the vertex from scratch because we need to modify the weight.
    // the weight must be the larger one, otherwise we may have outliers which are not real outliers

    if(bFoundDuplicate){
      // create backup track containers from main vertex
      std::vector<TrackBaseRef > tracks_;
      std::vector<Track> refittedTracks_;
      std::vector<float> weights_;
      for(reco::Vertex::trackRef_iterator it = coll[i].vert.tracks_begin(); it!=coll[i].vert.tracks_end(); ++it) {
       tracks_.push_back( *it);
       refittedTracks_.push_back( coll[i].vert.refittedTrack(*it));
       weights_.push_back( coll[i].vert.trackWeight(*it) );
      }
      // delete tracks and add all tracks back, and check in which vertex the weight is larger
      coll[i].vert.removeTracks();
      std::vector<Track>::iterator it2 = refittedTracks_.begin();
      std::vector<float>::iterator it3 = weights_.begin();
      for(reco::Vertex::trackRef_iterator it = tracks_.begin(); it!=tracks_.end(); ++it, ++it2, ++it3){
       float weight = *it3;
       float weight2= sv2.trackWeight(*it);
       Track refittedTrackWithLargerWeight = *it2;
       if( weight2 >weight) {
         weight = weight2;
         refittedTrackWithLargerWeight = sv2.refittedTrack(*it);
       }
       coll[i].vert.add(*it , refittedTrackWithLargerWeight  , weight);
      }
    }

    // remove the second vertex from the collection
    coll.erase( coll.begin() + k  );
  }

}

template<>
void BtoCharmDecayVertexMergerT<reco::VertexCompositePtrCandidate>::resolveBtoDchain(std::vector<VertexProxy>& coll, unsigned int i, unsigned int k){
  using namespace reco;

  Candidate::LorentzVector vtx1P4 = vtxP4(coll[i].vert);
  Candidate::LorentzVector vtx2P4 = vtxP4(coll[k].vert);
  GlobalVector momentum1 = GlobalVector(vtx1P4.X(), vtx1P4.Y(), vtx1P4.Z());
  GlobalVector momentum2 = GlobalVector(vtx2P4.X(), vtx2P4.Y(), vtx2P4.Z());

  SecondaryVertex sv1(pv, coll[i].vert, momentum1 , true);
  SecondaryVertex sv2(pv, coll[k].vert, momentum2 , true);


  // find out which one is near and far
  SecondaryVertex svNear = sv1;
  SecondaryVertex svFar  = sv2;
  GlobalVector momentumNear  = momentum1;
  GlobalVector momentumFar   = momentum2;

  // swap if it is the other way around
  if(sv1.dist3d().value() >= sv2.dist3d().value()){
    svNear = sv2;
    svFar = sv1;
    momentumNear = momentum2;
    momentumFar = momentum1;
  }
  GlobalVector nearToFar = flightDirection( svNear, svFar);
  GlobalVector pvToNear  = flightDirection( pv, svNear);
  GlobalVector pvToFar   = flightDirection( pv, svFar);

  double cosPA =  nearToFar.dot(momentumFar) / momentumFar.mag()/ nearToFar.mag();
  double cosa  =  pvToNear. dot(momentumFar) / pvToNear.mag()   / momentumFar.mag();
  double ptrel = sqrt(1.0 - cosa*cosa)* momentumFar.mag();

  double vertexDeltaR = Geom::deltaR(pvToNear, pvToFar);

  // create a set of all tracks from both vertices, avoid double counting by using a std::set<>
  std::set<reco::CandidatePtr> trackrefs;
  // first vertex
  for(size_t i=0; i < sv1.numberOfSourceCandidatePtrs(); ++i)
      trackrefs.insert(sv1.daughterPtr(i));
  // second vertex
  for(size_t i=0; i < sv2.numberOfSourceCandidatePtrs(); ++i)
      trackrefs.insert(sv2.daughterPtr(i));

  // now calculate one LorentzVector from the track momenta
  Candidate::LorentzVector mother;
  for(std::set<reco::CandidatePtr>::const_iterator it = trackrefs.begin(); it!= trackrefs.end(); ++it){
    Candidate::LorentzVector temp ( (*it)->px(),(*it)->py(),(*it)->pz(), ParticleMasses::piPlus );
    mother += temp;
  }

  //   check if criteria for merging are fulfilled
  if(vertexDeltaR<maxDRForUnique && mother.M()<maxvecSumIMCUTForUnique && cosPA>minCosPAtomerge && std::abs(ptrel)<maxPtreltomerge ) {

    // add tracks of second vertex which are missing to first vertex
    // loop over the second
    const std::vector<reco::CandidatePtr> & tracks1 = sv1.daughterPtrVector();
    const std::vector<reco::CandidatePtr> & tracks2 = sv2.daughterPtrVector();
    for(std::vector<reco::CandidatePtr>::const_iterator ti = tracks2.begin(); ti!=tracks2.end(); ++ti){
      std::vector<reco::CandidatePtr>::const_iterator it = find(tracks1.begin(), tracks1.end(), *ti);
      if (it==tracks1.end()) {
        coll[i].vert.addDaughter( *ti );
        coll[i].vert.setP4( (*ti)->p4() + coll[i].vert.p4() );
      }
    }

    // remove the second vertex from the collection
    coll.erase( coll.begin() + k  );
  }
}


// define specific instances of the templated BtoCharmDecayVertexMergerT class
typedef BtoCharmDecayVertexMergerT<reco::Vertex>                      BtoCharmDecayVertexMerger;
typedef BtoCharmDecayVertexMergerT<reco::VertexCompositePtrCandidate> CandidateBtoCharmDecayVertexMerger;

// define plugins
DEFINE_FWK_MODULE(BtoCharmDecayVertexMerger);
DEFINE_FWK_MODULE(CandidateBtoCharmDecayVertexMerger);
