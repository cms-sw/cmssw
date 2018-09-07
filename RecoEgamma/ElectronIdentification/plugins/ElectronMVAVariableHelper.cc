#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoEgamma/EgammaTools/interface/MultiToken.h"
#include "RecoEgamma/EgammaTools/interface/Utils.h"

#include <TMath.h>

class ElectronMVAVariableHelper : public edm::stream::EDProducer<> {
 public:

  explicit ElectronMVAVariableHelper(const edm::ParameterSet & iConfig);
  ~ElectronMVAVariableHelper() override ;

  void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:

  // for AOD and MiniAOD case
  MultiTokenT<edm::View<reco::GsfElectron>> electronsToken_;
  MultiTokenT<reco::VertexCollection>       vtxToken_;
  MultiTokenT<reco::ConversionCollection>   conversionsToken_;
  edm::EDGetTokenT<reco::BeamSpot>          beamSpotToken_;
};

ElectronMVAVariableHelper::ElectronMVAVariableHelper(const edm::ParameterSet & iConfig)
  : electronsToken_  (                 consumesCollector(), iConfig, "src"             , "srcMiniAOD")
  , vtxToken_        (electronsToken_, consumesCollector(), iConfig, "vertexCollection", "vertexCollectionMiniAOD")
  , conversionsToken_(electronsToken_, consumesCollector(), iConfig, "conversions"     , "conversionsMiniAOD")
  , beamSpotToken_   (consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot")))
{
  produces<edm::ValueMap<float>>("convVtxFitProb");
  produces<edm::ValueMap<float>>("kfhits");
  produces<edm::ValueMap<float>>("kfchi2");
}

ElectronMVAVariableHelper::~ElectronMVAVariableHelper()
{}

void ElectronMVAVariableHelper::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {

  // get Handles
  auto electrons      = electronsToken_.getValidHandle(iEvent);
  auto vtxH           = vtxToken_.getValidHandle(iEvent);
  auto conversions    = conversionsToken_.getValidHandle(iEvent);

  edm::Handle<reco::BeamSpot> beamSpotHandle; 
  iEvent.getByToken(beamSpotToken_, beamSpotHandle);

  const reco::VertexRef vtx(vtxH, 0);
  const reco::BeamSpot* beamSpot = &*(beamSpotHandle.product());

  // prepare vector for output
  std::vector<float> convVtxFitProbVals;
  std::vector<float> kfhitsVals;
  std::vector<float> kfchi2Vals;

  for (auto const& ele : *electrons) {

      // Conversion vertex fit
      reco::ConversionRef convRef = ConversionTools::matchedConversion(ele, conversions, beamSpot->position());

      float convVtxFitProb = -1.;
      if(!convRef.isNull()) {
          const reco::Vertex &vtx = convRef.get()->conversionVertex();
          if (vtx.isValid()) {
              convVtxFitProb = TMath::Prob( vtx.chi2(),  vtx.ndof());
          }
      }

      convVtxFitProbVals.push_back(convVtxFitProb);

      // kf track related variables
      bool validKf=false;
      reco::TrackRef trackRef = ele.closestCtfTrackRef();
      validKf = trackRef.isAvailable();
      validKf &= trackRef.isNonnull();
      float kfchi2 = validKf ? trackRef->normalizedChi2() : 0 ; //ielectron->track()->normalizedChi2() : 0 ;
      float kfhits = validKf ? trackRef->hitPattern().trackerLayersWithMeasurement() : -1. ;

      kfchi2Vals.push_back(kfchi2);
      kfhitsVals.push_back(kfhits);

  }

  // convert into ValueMap and store
  writeValueMap(iEvent, electrons, kfchi2Vals, "kfchi2" );
  writeValueMap(iEvent, electrons, kfhitsVals, "kfhits" );
  writeValueMap(iEvent, electrons, convVtxFitProbVals, "convVtxFitProb" );
}

void ElectronMVAVariableHelper::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // electronMVAVariableHelper
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("conversions",             edm::InputTag("allConversions"));
  desc.add<edm::InputTag>("src",                     edm::InputTag("gedGsfElectrons"));
  desc.add<edm::InputTag>("srcMiniAOD",              edm::InputTag("slimmedElectrons","","@skipCurrentProcess"));
  desc.add<edm::InputTag>("beamSpot",                edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("conversionsMiniAOD",      edm::InputTag("reducedEgamma","reducedConversions"));
  desc.add<edm::InputTag>("vertexCollection",        edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("vertexCollectionMiniAOD", edm::InputTag("offlineSlimmedPrimaryVertices"));
  descriptions.add("electronMVAVariableHelper", desc);
}

DEFINE_FWK_MODULE(ElectronMVAVariableHelper);
