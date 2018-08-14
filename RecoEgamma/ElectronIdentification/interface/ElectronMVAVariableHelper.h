#ifndef ELECTRONMVAVARIABLEHELPER_H
#define ELECTRONMVAVARIABLEHELPER_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/PatCandidates/interface/Electron.h"

#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

#include <TMath.h>

typedef edm::View<reco::Candidate> CandView;

template <class ParticleType>
class ElectronMVAVariableHelper : public edm::stream::EDProducer<> {
 public:

  explicit ElectronMVAVariableHelper(const edm::ParameterSet & iConfig);
  ~ElectronMVAVariableHelper() override ;

  void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) override;

private:
  template<typename T>
  void writeValueMap(edm::Event &iEvent,
            const edm::Handle<edm::View<ParticleType> > & handle,
            const std::vector<T> & values,
            const std::string    & label) const ;

  // for AOD case
  const edm::EDGetTokenT<edm::View<ParticleType> > electronsToken_;
  const edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  const edm::EDGetTokenT<reco::ConversionCollection> conversionsToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;

  // for miniAOD case
  const edm::EDGetTokenT<edm::View<ParticleType> > electronsTokenMiniAOD_;
  const edm::EDGetTokenT<reco::VertexCollection> vtxTokenMiniAOD_;
  const edm::EDGetTokenT<reco::ConversionCollection> conversionsTokenMiniAOD_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotTokenMiniAOD_;
};

template<class ParticleType>
ElectronMVAVariableHelper<ParticleType>::ElectronMVAVariableHelper(const edm::ParameterSet & iConfig) :
  electronsToken_(consumes<edm::View<ParticleType> >(iConfig.getParameter<edm::InputTag>("src"))),
  vtxToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexCollection"))),
  conversionsToken_(consumes<reco::ConversionCollection>(iConfig.getParameter<edm::InputTag>("conversions"))),
  beamSpotToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
  electronsTokenMiniAOD_(consumes<edm::View<ParticleType> >(iConfig.getParameter<edm::InputTag>("srcMiniAOD"))),
  vtxTokenMiniAOD_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexCollectionMiniAOD"))),
  conversionsTokenMiniAOD_(consumes<reco::ConversionCollection>(iConfig.getParameter<edm::InputTag>("conversionsMiniAOD"))),
  beamSpotTokenMiniAOD_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotMiniAOD"))) {

  produces<edm::ValueMap<float>>("convVtxFitProb");
  produces<edm::ValueMap<float>>("kfhits");
  produces<edm::ValueMap<float>>("kfchi2");
}

template<class ParticleType>
ElectronMVAVariableHelper<ParticleType>::~ElectronMVAVariableHelper()
{}

template<class ParticleType>
void ElectronMVAVariableHelper<ParticleType>::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {

  // read input
  edm::Handle<edm::View<ParticleType> > electrons;
  edm::Handle<reco::VertexCollection> vtxH;
  edm::Handle<reco::ConversionCollection> conversions;
  edm::Handle<reco::BeamSpot> beamSpotHandle;

  bool isAOD = true;
  // Retrieve the collection of particles from the event.
  // If we fail to retrieve the collection with the standard AOD
  // name, we next look for the one with the stndard miniAOD name.
  iEvent.getByToken(electronsToken_, electrons);
  if( !electrons.isValid() ){
    isAOD = false;
    iEvent.getByToken(electronsTokenMiniAOD_,electrons);
    if( !electrons.isValid() )
      throw cms::Exception(" Collection not found: ") << " failed to find a standard AOD or miniAOD particle collection " << std::endl;
  }

  if (isAOD) {
      iEvent.getByToken(vtxToken_, vtxH);
      iEvent.getByToken(conversionsToken_, conversions);
      iEvent.getByToken(beamSpotToken_, beamSpotHandle);
  } else {
      iEvent.getByToken(vtxTokenMiniAOD_, vtxH);
      iEvent.getByToken(conversionsTokenMiniAOD_, conversions);
      iEvent.getByToken(beamSpotTokenMiniAOD_, beamSpotHandle);
  }

  // Make sure everything is retrieved successfully
  if(! (beamSpotHandle.isValid() && conversions.isValid() && vtxH.isValid() ) ) {
    throw cms::Exception("MVA failure: ")
      << "Failed to retrieve event content needed for this MVA"
      << std::endl
      << "Check python MVA configuration file."
      << std::endl;
  }

  const reco::VertexRef vtx(vtxH, 0);
  const reco::BeamSpot* beamSpot = &*(beamSpotHandle.product());

  // prepare vector for output
  std::vector<float> convVtxFitProbVals;
  std::vector<float> kfhitsVals;
  std::vector<float> kfchi2Vals;

  for (size_t i = 0; i < electrons->size(); ++i){
      auto iCand = electrons->ptrAt(i);

      // Conversion vertex fit
      reco::ConversionRef convRef = ConversionTools::matchedConversion(*iCand, conversions, beamSpot->position());

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
      reco::TrackRef trackRef = iCand->closestCtfTrackRef();
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

template<class ParticleType> template<typename T>
void ElectronMVAVariableHelper<ParticleType>::writeValueMap(edm::Event &iEvent,
                                                      const edm::Handle<edm::View<ParticleType> > & handle,
                                                      const std::vector<T> & values,
                                                      const std::string    & label) const
{
  auto valMap = std::make_unique<edm::ValueMap<T>>();
  typename edm::ValueMap<T>::Filler filler(*valMap);
  filler.insert(handle, values.begin(), values.end());
  filler.fill();
  iEvent.put(std::move(valMap), label);
}

#endif
