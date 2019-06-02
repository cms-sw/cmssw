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
  EventVtxInfoNtupleDumper(const edm::ParameterSet &);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;
  edm::EDGetTokenT<reco::VertexCollection> primaryVerticesToken_;
};

EventVtxInfoNtupleDumper::EventVtxInfoNtupleDumper(const ParameterSet &cfg)
    : primaryVerticesToken_(consumes<reco::VertexCollection>(cfg.getParameter<InputTag>("primaryVertices"))) {
  produces<int>("numPV").setBranchAlias("numPV");
  produces<int>("nTrkPV").setBranchAlias("nTrkPV");
  produces<float>("chi2PV").setBranchAlias("chi2PV");
  produces<float>("ndofPV").setBranchAlias("ndofPV");
  produces<float>("zPV").setBranchAlias("zPV");
  produces<float>("rhoPV").setBranchAlias("rhoPV");
  //  produces<std::vector< unsigned int > >( "nTrkPV" ).setBranchAlias( "nTrkPV" );
  //  produces<std::vector< float > >( "chi2PV" ).setBranchAlias( "chi2PV" );
  //  produces<std::vector< float > >( "ndofPV" ).setBranchAlias( "ndofPV" );
}

void EventVtxInfoNtupleDumper::produce(Event &evt, const EventSetup &) {
  Handle<reco::VertexCollection> primaryVertices;  // Collection of primary Vertices
  evt.getByToken(primaryVerticesToken_, primaryVertices);
  unique_ptr<int> nVtxs(new int);
  unique_ptr<int> nTrkVtx(new int);
  unique_ptr<float> chi2Vtx(new float);
  unique_ptr<float> ndofVtx(new float);
  unique_ptr<float> zVtx(new float);
  unique_ptr<float> rhoVtx(new float);
  //  unique_ptr< vector< unsigned int > > nTrkVtx( new vector< unsigned int > );
  //  unique_ptr< vector< float > > chi2Vtx( new vector< float > );
  //  unique_ptr< vector< float > > ndofVtx( new vector< float > );

  const reco::Vertex &pv = (*primaryVertices)[0];

  *nVtxs = -1;
  *nTrkVtx = -1;
  *chi2Vtx = -1.0;
  *ndofVtx = -1.0;
  *zVtx = -1000;
  *rhoVtx = -1000;
  if (!(pv.isFake())) {
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
  evt.put(std::move(nVtxs), "numPV");
  evt.put(std::move(nTrkVtx), "nTrkPV");
  evt.put(std::move(chi2Vtx), "chi2PV");
  evt.put(std::move(ndofVtx), "ndofPV");
  evt.put(std::move(zVtx), "zPV");
  evt.put(std::move(rhoVtx), "rhoPV");
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(EventVtxInfoNtupleDumper);
