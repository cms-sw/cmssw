// this code is partially taken from BCandidateProducer of L. Wehrli

// first revised prototype version for CMSSW 29.Aug.2012

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoBTag/SecondaryVertex/interface/SecondaryVertex.h"

#include "DataFormats/GeometryVector/interface/VectorUtil.h"

#include <algorithm>

class BtoCharmDecayVertexMerger : public edm::EDProducer {
    public:
	BtoCharmDecayVertexMerger(const edm::ParameterSet &params);

	virtual void produce(edm::Event &event, const edm::EventSetup &es);

    private:

 
  edm::InputTag				primaryVertexCollection;
  edm::InputTag				secondaryVertexCollection;
  reco::Vertex pv;
  //	double					maxFraction;
  //	double					minSignificance;

  double minDRForUnique, vecSumIMCUTForUnique, minCosPAtomerge, maxPtreltomerge; 

  // a vertex proxy for sorting using stl
  struct vertexProxy{
    reco::Vertex vert; 
    double invm;
    size_t ntracks;
  };

  // comparison operator for vertexProxy, used in sorting
  friend bool operator<(vertexProxy v1, vertexProxy v2){
    if(v1.ntracks>2 && v2.ntracks<3) return true;
    if(v1.ntracks<3 && v2.ntracks>2) return false;
    return (v1.invm>v2.invm);
  }

  void resolveBtoDchain(std::vector<vertexProxy>& coll, unsigned int i, unsigned int k);
  GlobalVector flightDirection(reco::Vertex &pv, reco::Vertex &sv);
};



BtoCharmDecayVertexMerger::BtoCharmDecayVertexMerger(const edm::ParameterSet &params) :
	primaryVertexCollection(params.getParameter<edm::InputTag>("primaryVertices")),
	secondaryVertexCollection(params.getParameter<edm::InputTag>("secondaryVertices")),
	minDRForUnique(params.getUntrackedParameter<double>("minDRUnique",0.4)),
	vecSumIMCUTForUnique(params.getUntrackedParameter<double>("minvecSumIMifsmallDRUnique",5.5)), 
	minCosPAtomerge(params.getUntrackedParameter<double>("minCosPAtomerge",0.99)),
	maxPtreltomerge(params.getUntrackedParameter<double>("maxPtreltomerge",7777.0))
						    //	maxFraction(params.getParameter<double>("maxFraction")),
						    //	minSignificance(params.getParameter<double>("minSignificance"))
{
	produces<reco::VertexCollection>();
}
//-----------------------
void BtoCharmDecayVertexMerger::produce(edm::Event &iEvent, const edm::EventSetup &iSetup){

  using namespace reco;
  // PV
  edm::Handle<reco::VertexCollection> PVcoll;
  iEvent.getByLabel(primaryVertexCollection, PVcoll);

  if(PVcoll->size()!=0) {

  const reco::VertexCollection pvc = *( PVcoll.product());
  pv = pvc[0];

  // get the IVF collection
  edm::Handle<VertexCollection> secondaryVertices;
  iEvent.getByLabel(secondaryVertexCollection, secondaryVertices);
  
 

  //loop over vertices,  fill into own collection for sorting 
  std::vector<vertexProxy> vertexProxyColl; 
  for(std::vector<reco::Vertex>::const_iterator sv = secondaryVertices->begin();
      sv != secondaryVertices->end(); ++sv) {
    vertexProxy vtx = {*sv,(*sv).p4().M(),(*sv).tracksSize()};
    vertexProxyColl.push_back( vtx );
  }
  
  // sort the vertices by mass and track multiplicity
  sort( vertexProxyColl.begin(), vertexProxyColl.end()); 
  
  
  // loop forward over all vertices
  for(unsigned int iVtx=0; iVtx < vertexProxyColl.size(); iVtx++){

    // nested loop backwards (in order to start from light masses)
    // check all vertices against each other for B->D chain
    for(unsigned int kVtx=vertexProxyColl.size()-1; kVtx>iVtx; kVtx--){
      // remove D vertices from the collection and add the tracks to the original one
      resolveBtoDchain(vertexProxyColl, iVtx, kVtx); 
    }
  }

  // now create new vertex collection and add to event
  VertexCollection *bvertices = new VertexCollection();
  for(std::vector<vertexProxy>::iterator it=vertexProxyColl.begin(); it!=vertexProxyColl.end(); it++) bvertices->push_back((*it).vert); 
  std::auto_ptr<VertexCollection> bvertColl(bvertices);
  iEvent.put(bvertColl);
  }
  else{
    std::auto_ptr<VertexCollection> bvertCollEmpty(new VertexCollection);
    iEvent.put(bvertCollEmpty);
  }
}


//------------------------------------
void BtoCharmDecayVertexMerger::resolveBtoDchain(std::vector<vertexProxy>& coll, unsigned int i, unsigned int k){
  using namespace reco;

  GlobalVector momentum1 = GlobalVector(coll[i].vert.p4().X(), coll[i].vert.p4().Y(), coll[i].vert.p4().Z());
  GlobalVector momentum2 = GlobalVector(coll[k].vert.p4().X(), coll[k].vert.p4().Y(), coll[k].vert.p4().Z());

  reco::SecondaryVertex sv1(pv, coll[i].vert, momentum1 , true);
  reco::SecondaryVertex sv2(pv, coll[k].vert, momentum2 , true);

 
  // find out which one is near and far
  reco::SecondaryVertex svNear = sv1;
  reco::SecondaryVertex svFar  = sv2;
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

  double cosPA =  nearToFar.dot(momentumFar) / momentumFar.mag()/ nearToFar.mag();  
  double cosa  =  pvToNear. dot(momentumFar) / pvToNear.mag()   / momentumFar.mag(); 
  double ptrel = sqrt(1.0 - cosa*cosa)* momentumFar.mag();  
  
  double vertexDeltaR = Geom::deltaR(flightDirection(pv, sv1), flightDirection(pv, sv2) );

  // create a set of all tracks from both vertices, avoid double counting by using a std::set<>
  std::set<reco::TrackRef> trackrefs;
  // first vertex
  for(reco::Vertex::trackRef_iterator ti = sv1.tracks_begin(); ti!=sv1.tracks_end(); ti++){
    if(sv1.trackWeight(*ti)>0.5){
      reco::TrackRef t = ti->castTo<reco::TrackRef>();
      trackrefs.insert(t);
    }
  }
  // second vertex
  for(reco::Vertex::trackRef_iterator ti = sv2.tracks_begin(); ti!=sv2.tracks_end(); ti++){
    if(sv2.trackWeight(*ti)>0.5){
      reco::TrackRef t = ti->castTo<reco::TrackRef>();
      trackrefs.insert(t);
    }
  }
  
  // now calculate one LorentzVector from the track momenta
  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > mother;
  for(std::set<reco::TrackRef>::const_iterator it = trackrefs.begin(); it!= trackrefs.end(); it++){
    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> >  temp ( (*it)->px(),(*it)->py(),(*it)->pz(), 0.13957 );
    mother += temp;
  }
 

  //   check if criteria for merging are fulfilled
  if(vertexDeltaR<minDRForUnique && mother.M()<vecSumIMCUTForUnique && cosPA>minCosPAtomerge && fabs(ptrel)<maxPtreltomerge ) {
    
    
    // add tracks of second vertex which are missing to first vertex
    // loop over the second
    bool bFoundDuplicate=false;
    for(reco::Vertex::trackRef_iterator ti = sv2.tracks_begin(); ti!=sv2.tracks_end(); ti++){
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
      for(reco::Vertex::trackRef_iterator it = coll[i].vert.tracks_begin(); it!=coll[i].vert.tracks_end(); it++) {
	tracks_.push_back( *it);
	refittedTracks_.push_back( coll[i].vert.refittedTrack(*it));
	weights_.push_back( coll[i].vert.trackWeight(*it) );
      }
      // delete tracks and add all tracks back, and check in which vertex the weight is larger
      coll[i].vert.removeTracks();
      std::vector<Track>::iterator it2 = refittedTracks_.begin();
      std::vector<float>::iterator it3 = weights_.begin();
      for(reco::Vertex::trackRef_iterator it = tracks_.begin(); it!=tracks_.end(); it++, it2++, it3++){
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
//-------------


GlobalVector
BtoCharmDecayVertexMerger::flightDirection(reco::Vertex &pv, reco::Vertex &sv){
  GlobalVector res(sv.position().X() - pv.position().X(),
                    sv.position().Y() - pv.position().Y(),
                    sv.position().Z() - pv.position().Z());
  return res;
}



DEFINE_FWK_MODULE(BtoCharmDecayVertexMerger);
