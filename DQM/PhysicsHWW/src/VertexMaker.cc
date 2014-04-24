#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQM/PhysicsHWW/interface/VertexMaker.h"

typedef math::XYZTLorentzVectorF LorentzVector;

VertexMaker::VertexMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector iCollector){

  thePVCollection_       = iCollector.consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertexInputTag"));

}

void VertexMaker::SetVars(HWW& hww, const edm::Event& iEvent, const edm::EventSetup& iSetup){

  hww.Load_vtxs_position();
  hww.Load_vtxs_xError();
  hww.Load_vtxs_yError();
  hww.Load_vtxs_zError();
  hww.Load_vtxs_ndof();
  hww.Load_vtxs_isFake();
  hww.Load_vtxs_sumpt();
  hww.Load_vtxs_covMatrix();

  bool validToken;

  // get the primary vertices
  edm::Handle<reco::VertexCollection> vertexHandle;
  validToken = iEvent.getByToken(thePVCollection_, vertexHandle); 
  if(!validToken) return;
  const reco::VertexCollection *vertexCollection = vertexHandle.product();

  unsigned int index = 0;
  const unsigned int covMatrix_dim = 3;

  for (reco::VertexCollection::const_iterator vtx = vertexCollection->begin(); vtx != vertexCollection->end(); ++vtx, ++index) {
    hww.vtxs_position()         .push_back( LorentzVector( vtx->position().x(), vtx->position().y(), vtx->position().z(), 0 ) );
    hww.vtxs_xError()           .push_back( vtx->xError()            );
    hww.vtxs_yError()           .push_back( vtx->yError()            );
    hww.vtxs_zError()           .push_back( vtx->zError()            );
    hww.vtxs_ndof()             .push_back( vtx->ndof()              );
    hww.vtxs_isFake()           .push_back( vtx->isFake()            );
    double sumpt = 0;
    for (reco::Vertex::trackRef_iterator i = vtx->tracks_begin(); i != vtx->tracks_end(); ++i) sumpt += (*i)->pt();

    hww.vtxs_sumpt().push_back(sumpt);

    std::vector<float> temp_vec;
    temp_vec.clear();

    for( unsigned int i = 0; i < covMatrix_dim; i++ ) {
      for( unsigned int j = 0; j < covMatrix_dim; j++ ) {
        temp_vec.push_back( vtx->covariance(i, j) );
      }
    }

    hww.vtxs_covMatrix().push_back( temp_vec );
    
  }

}
