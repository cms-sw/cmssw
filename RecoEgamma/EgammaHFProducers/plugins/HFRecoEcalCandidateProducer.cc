//
// Package:    EgammaHFProducers
// Class:      HFRecoEcalCandidateProducers
//
/**\class HFRecoEcalCandidateProducers.cc  
*/
//
// Original Author:  Kevin Klapoetke University of Minnesota
//         Created:  Wed 26 Sept 2007
// $Id:
//
//

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/HFEMClusterShape.h"
#include "DataFormats/EgammaReco/interface/HFEMClusterShapeAssociation.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "HFRecoEcalCandidateAlgo.h"
#include "HFValueStruct.h"

#include <vector>
#include <memory>

class HFRecoEcalCandidateProducer : public edm::stream::EDProducer<> {
public:
  explicit HFRecoEcalCandidateProducer(edm::ParameterSet const& conf);
  void produce(edm::Event& e, edm::EventSetup const& iSetup) override;

private:
  std::vector<double> defaultDB_;
  edm::EDGetToken hfclustersSC_, hfclustersHFEM_, vertices_;
  int HFDBversion_;
  std::vector<double> HFDBvector_;
  bool doPU_;
  double Cut2D_;
  double defaultSlope2D_;
  reco::HFValueStruct hfvars_;
  HFRecoEcalCandidateAlgo algo_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HFRecoEcalCandidateProducer);

HFRecoEcalCandidateProducer::HFRecoEcalCandidateProducer(edm::ParameterSet const& conf)
    : defaultDB_(std::vector<double>()),
      hfclustersSC_(consumes<reco::SuperClusterCollection>(conf.getParameter<edm::InputTag>("hfclusters"))),
      hfclustersHFEM_(
          consumes<reco::HFEMClusterShapeAssociationCollection>(conf.getParameter<edm::InputTag>("hfclusters"))),

      HFDBversion_(conf.existsAs<int>("HFDBversion") ? conf.getParameter<int>("HFDBversion") : 99),  //do nothing
      HFDBvector_(conf.existsAs<std::vector<double> >("HFDBvector")
                      ? conf.getParameter<std::vector<double> >("HFDBvector")
                      : defaultDB_),
      doPU_(false),
      Cut2D_(conf.getParameter<double>("intercept2DCut")),
      defaultSlope2D_(
          (Cut2D_ <= 0.83)
              ? (0.475)
              : ((Cut2D_ > 0.83 && Cut2D_ <= 0.9) ? (0.275) : (0.2))),  //fix for hlt unable to add slope variable now
      hfvars_(HFDBversion_, HFDBvector_),
      algo_(conf.existsAs<bool>("Correct") ? conf.getParameter<bool>("Correct") : true,
            conf.getParameter<double>("e9e25Cut"),
            conf.getParameter<double>("intercept2DCut"),
            conf.existsAs<double>("intercept2DSlope") ? conf.getParameter<double>("intercept2DSlope") : defaultSlope2D_,
            conf.getParameter<std::vector<double> >("e1e9Cut"),
            conf.getParameter<std::vector<double> >("eCOREe9Cut"),
            conf.getParameter<std::vector<double> >("eSeLCut"),
            hfvars_) {
  if (conf.existsAs<edm::InputTag>("VertexCollection")) {
    vertices_ = consumes<reco::VertexCollection>(conf.getParameter<edm::InputTag>("VertexCollection"));
  } else
    vertices_ = consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));

  produces<reco::RecoEcalCandidateCollection>();
}

void HFRecoEcalCandidateProducer::produce(edm::Event& e, edm::EventSetup const& iSetup) {
  edm::Handle<reco::SuperClusterCollection> super_clus;
  edm::Handle<reco::HFEMClusterShapeAssociationCollection> hf_assoc;

  e.getByToken(hfclustersSC_, super_clus);
  e.getByToken(hfclustersHFEM_, hf_assoc);

  int nvertex = 0;
  if (HFDBversion_ != 99) {
    edm::Handle<reco::VertexCollection> pvHandle;
    e.getByToken(vertices_, pvHandle);
    const reco::VertexCollection& vertices = *pvHandle.product();
    static const int minNDOF = 4;
    static const double maxAbsZ = 15.0;
    static const double maxd0 = 2.0;

    //count verticies

    for (reco::VertexCollection::const_iterator vit = vertices.begin(); vit != vertices.end(); ++vit) {
      if (vit->ndof() > minNDOF && ((maxAbsZ <= 0) || fabs(vit->z()) <= maxAbsZ) &&
          ((maxd0 <= 0) || fabs(vit->position().rho()) <= maxd0))
        nvertex++;
    }
  } else {
    nvertex = 1;
  }

  // create return data
  auto retdata1 = std::make_unique<reco::RecoEcalCandidateCollection>();

  algo_.produce(super_clus, *hf_assoc, *retdata1, nvertex);

  e.put(std::move(retdata1));
}
