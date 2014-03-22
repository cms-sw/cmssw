#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <vector>

using namespace edm;
using namespace std;
using namespace reco;


class EventVtxInfoNtupleDumper : public edm::EDProducer {
public:
  EventVtxInfoNtupleDumper( const edm::ParameterSet & );

private:
  void produce( edm::Event &, const edm::EventSetup & ) override;
  edm::EDGetTokenT<reco::VertexCollection> primaryVerticesToken_;

};

EventVtxInfoNtupleDumper::EventVtxInfoNtupleDumper( const ParameterSet & cfg ) :
  primaryVerticesToken_(consumes<reco::VertexCollection>(cfg.getParameter<InputTag>("primaryVertices")) ){
  produces<int>( "numPV" ).setBranchAlias( "numPV" );
  produces<int>( "nTrkPV" ).setBranchAlias( "nTrkPV" );
  produces<float>( "chi2PV" ).setBranchAlias( "chi2PV" );
  produces<float>( "ndofPV" ).setBranchAlias( "ndofPV" );
  produces<float>( "zPV" ).setBranchAlias( "zPV" );
  produces<float>( "rhoPV" ).setBranchAlias( "rhoPV" );
  //  produces<std::vector< unsigned int > >( "nTrkPV" ).setBranchAlias( "nTrkPV" );
  //  produces<std::vector< float > >( "chi2PV" ).setBranchAlias( "chi2PV" );
  //  produces<std::vector< float > >( "ndofPV" ).setBranchAlias( "ndofPV" );
}



void EventVtxInfoNtupleDumper::produce( Event & evt, const EventSetup & ) {

  Handle<reco::VertexCollection> primaryVertices;  // Collection of primary Vertices
  evt.getByToken(primaryVerticesToken_, primaryVertices);
  auto_ptr<int> nVtxs( new int );
  auto_ptr<int> nTrkVtx( new int );
  auto_ptr<float> chi2Vtx( new float );
  auto_ptr<float> ndofVtx( new float );
  auto_ptr<float> zVtx( new float );
  auto_ptr<float> rhoVtx( new float );
  //  auto_ptr< vector< unsigned int > > nTrkVtx( new vector< unsigned int > );
  //  auto_ptr< vector< float > > chi2Vtx( new vector< float > );
  //  auto_ptr< vector< float > > ndofVtx( new vector< float > );

  const reco::Vertex &pv = (*primaryVertices)[0];

  *nVtxs = -1;
  *nTrkVtx = -1;
  *chi2Vtx = -1.0;
  *ndofVtx = -1.0;
  *zVtx = -1000;
  *rhoVtx = -1000;
  if( !(pv.isFake()) ) {
    *nVtxs = primaryVertices->size();
    *nTrkVtx = pv.tracksSize();
    *chi2Vtx = pv.chi2();
    *ndofVtx = pv.ndof();
    *zVtx = pv.z();
    *rhoVtx = pv.position().Rho();
  }
  //  nTrkVtx->push_back(pv.tracksSize());
  //  chi2Vtx->push_back(pv.chi2());
  //  ndofVtx->push_back(pv.ndof());
  evt.put( nVtxs, "numPV" );
  evt.put( nTrkVtx, "nTrkPV" );
  evt.put( chi2Vtx, "chi2PV" );
  evt.put( ndofVtx, "ndofPV" );
  evt.put( zVtx, "zPV" );
  evt.put( rhoVtx, "rhoPV" );

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( EventVtxInfoNtupleDumper );

