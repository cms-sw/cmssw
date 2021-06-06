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
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "HFRecoEcalCandidateAlgo.h"
#include "HFValueStruct.h"

#include <vector>
#include <memory>

class HLTHFRecoEcalCandidateProducer : public edm::global::EDProducer<> {
public:
  explicit HLTHFRecoEcalCandidateProducer(edm::ParameterSet const& conf);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

private:
  const edm::InputTag hfclusters_, vertices_;
  const int HFDBversion_;
  const std::vector<double> HFDBvector_;
  const double Cut2D_;
  const double defaultSlope2D_;
  const reco::HFValueStruct hfvars_;
  const HFRecoEcalCandidateAlgo algo_;
};

HLTHFRecoEcalCandidateProducer::HLTHFRecoEcalCandidateProducer(edm::ParameterSet const& conf)
    : hfclusters_(conf.getParameter<edm::InputTag>("hfclusters")),
      HFDBversion_(conf.existsAs<bool>("HFDBversion") ? conf.getParameter<int>("HFDBversion") : 99),  //do nothing
      HFDBvector_(conf.existsAs<bool>("HFDBvector") ? conf.getParameter<std::vector<double> >("HFDBvector")
                                                    : std::vector<double>{}),
      Cut2D_(conf.getParameter<double>("intercept2DCut")),
      defaultSlope2D_(
          (Cut2D_ <= 0.83)
              ? (0.475)
              : ((Cut2D_ > 0.83 && Cut2D_ <= 0.9) ? (0.275) : (0.2))),  //fix for hlt unable to add slope variable now
      hfvars_(HFDBversion_, HFDBvector_),
      algo_(conf.existsAs<bool>("Correct") ? conf.getParameter<bool>("Correct") : true,
            conf.getParameter<double>("e9e25Cut"),
            conf.getParameter<double>("intercept2DCut"),
            conf.existsAs<bool>("intercept2DSlope") ? conf.getParameter<double>("intercept2DSlope") : defaultSlope2D_,
            conf.getParameter<std::vector<double> >("e1e9Cut"),
            conf.getParameter<std::vector<double> >("eCOREe9Cut"),
            conf.getParameter<std::vector<double> >("eSeLCut"),
            hfvars_) {
  produces<reco::RecoEcalCandidateCollection>();
}

void HLTHFRecoEcalCandidateProducer::produce(edm::StreamID sid, edm::Event& e, edm::EventSetup const& iSetup) const {
  edm::Handle<reco::SuperClusterCollection> super_clus;
  edm::Handle<reco::HFEMClusterShapeAssociationCollection> hf_assoc;

  e.getByLabel(hfclusters_, super_clus);
  e.getByLabel(hfclusters_, hf_assoc);

  int nvertex = 1;

  // create return data
  auto retdata1 = std::make_unique<reco::RecoEcalCandidateCollection>();

  algo_.produce(super_clus, *hf_assoc, *retdata1, nvertex);

  e.put(std::move(retdata1));
}
