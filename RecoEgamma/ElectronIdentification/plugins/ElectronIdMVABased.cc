//
//
// Original Author:  Zablocki Jakub
//         Created:  Thu Feb  9 10:47:50 CST 2012
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimator.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
//
// class declaration
//

class ElectronIdMVABased : public edm::global::EDProducer<> {
public:
  explicit ElectronIdMVABased(const edm::ParameterSet&);
  ~ElectronIdMVABased() override {}

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<reco::VertexCollection> vertexToken;
  const edm::EDGetTokenT<reco::GsfElectronCollection> electronToken;
  const std::vector<std::string> mvaWeightFileEleID;
  const std::string path_mvaWeightFileEleID;
  const double thresholdBarrel;
  const double thresholdEndcap;
  const double thresholdIsoBarrel;
  const double thresholdIsoEndcap;

  const std::unique_ptr<const ElectronMVAEstimator> mvaID_;
};

// constructor
ElectronIdMVABased::ElectronIdMVABased(const edm::ParameterSet& iConfig)
  : vertexToken(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexTag")))
  , electronToken(consumes<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electronTag")))
  , thresholdBarrel   (iConfig.getParameter<double>("thresholdBarrel"))
  , thresholdEndcap   (iConfig.getParameter<double>("thresholdEndcap"))
  , thresholdIsoBarrel(iConfig.getParameter<double>("thresholdIsoDR03Barrel"))
  , thresholdIsoEndcap(iConfig.getParameter<double>("thresholdIsoDR03Endcap"))
  , mvaID_(new ElectronMVAEstimator(ElectronMVAEstimator::Configuration{
              .vweightsfiles = iConfig.getParameter<std::vector<std::string> >("HZZmvaWeightFile")}))
{
  produces<reco::GsfElectronCollection>();
}

// ------------ method called on each new Event  ------------
void ElectronIdMVABased::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{

  constexpr double etaEBEE = 1.485;

  auto mvaElectrons = std::make_unique<reco::GsfElectronCollection>();

  edm::Handle<reco::VertexCollection>  vertexCollection;
  iEvent.getByToken(vertexToken, vertexCollection);
  int nVtx = vertexCollection->size();

  edm::Handle<reco::GsfElectronCollection> egCollection;
  iEvent.getByToken(electronToken,egCollection);
  const reco::GsfElectronCollection egCandidates = (*egCollection.product());
  for ( reco::GsfElectronCollection::const_iterator egIter = egCandidates.begin(); egIter != egCandidates.end(); ++egIter) {
    double mvaVal = mvaID_->mva( *egIter, nVtx );
    double isoDr03 = egIter->dr03TkSumPt() + egIter->dr03EcalRecHitSumEt() + egIter->dr03HcalTowerSumEt();
    double eleEta = fabs(egIter->eta());
    if (eleEta <= etaEBEE && mvaVal > thresholdBarrel && isoDr03 < thresholdIsoBarrel) {
      mvaElectrons->push_back( *egIter );
      reco::GsfElectron::MvaOutput myMvaOutput;
      myMvaOutput.mva_Isolated = mvaVal;
      mvaElectrons->back().setMvaOutput(myMvaOutput);
    }
    else if (eleEta > etaEBEE && mvaVal > thresholdEndcap  && isoDr03 < thresholdIsoEndcap) {
      mvaElectrons->push_back( *egIter );
      reco::GsfElectron::MvaOutput myMvaOutput;
      myMvaOutput.mva_Isolated = mvaVal;
      mvaElectrons->back().setMvaOutput(myMvaOutput);
    }
  }

  iEvent.put(std::move(mvaElectrons));

}

//define this as a plug-in
DEFINE_FWK_MODULE(ElectronIdMVABased);
