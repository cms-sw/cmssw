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
  void produce( edm::Event &, const edm::EventSetup & );
  edm::InputTag primaryVertices_;

};

EventVtxInfoNtupleDumper::EventVtxInfoNtupleDumper( const ParameterSet & cfg ) : 
  primaryVertices_(cfg.getParameter<InputTag>("primaryVertices")) {
  produces<int>( "numPV" ).setBranchAlias( "numPV" );
  produces<std::vector< unsigned int > >( "nTrkPV" ).setBranchAlias( "nTrkPV" );
  produces<std::vector< float > >( "chi2PV" ).setBranchAlias( "chi2PV" );
  produces<std::vector< float > >( "ndofPV" ).setBranchAlias( "ndofPV" );
}



void EventVtxInfoNtupleDumper::produce( Event & evt, const EventSetup & ) {
  
  Handle<reco::VertexCollection> primaryVertices;  // Collection of primary Vertices
  evt.getByLabel(primaryVertices_, primaryVertices);
  auto_ptr<int> nVtxs( new int );
  auto_ptr< vector< unsigned int > > nTrkVtx( new vector< unsigned int > );
  auto_ptr< vector< float > > chi2Vtx( new vector< float > );
  auto_ptr< vector< float > > ndofVtx( new vector< float > );

  const reco::Vertex &pv = (*primaryVertices)[0];

  *nVtxs = -1;
  if( !(pv.isFake()) )
    *nVtxs = primaryVertices->size();

  nTrkVtx->push_back(pv.tracksSize());
  chi2Vtx->push_back(pv.chi2());
  ndofVtx->push_back(pv.ndof());
  evt.put( nVtxs, "numPV" );
  evt.put( nTrkVtx, "nTrkPV" );
  evt.put( chi2Vtx, "chi2PV" );
  evt.put( ndofVtx, "ndofPV" );

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( EventVtxInfoNtupleDumper );

