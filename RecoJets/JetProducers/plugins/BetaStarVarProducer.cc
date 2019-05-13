// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      BetaStarVarProducer
// 
/**\class BetaStarVarProducer BetaStarVarProducer.cc PhysicsTools/NanoAOD/plugins/BetaStarVarProducer.cc

 Description: This produces value maps to store CHS-related variables for JERC. 
              This includes the charged hadrons associated to CHS jets, 
              and those associated to PU that are within the CHS jet. 

 Implementation:
     This uses a ValueMap producer functionality, and 
     loops over the input candidates (usually PackedCandidates)
     that are associated to each jet, counting the candidates associated
     to the PV and those not. 
*/
//
// Original Author:  Sal Rappoccio
// Further editing:  Hannu Siikonen
//
//


#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/Common/interface/View.h"

#include "PhysicsTools/NanoAOD/interface/MatchingUtils.h"

template <typename T>
class BetaStarVarProducer : public edm::global::EDProducer<> {
public:
  explicit BetaStarVarProducer(const edm::ParameterSet &iConfig):
    srcJet_(consumes<edm::View<pat::Jet>>(iConfig.getParameter<edm::InputTag>("srcJet"))),
    srcPF_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("srcPF"))),
    maxDR_( iConfig.getParameter<double>("maxDR") )
  {
    produces<edm::ValueMap<float>>("chargedFromPV0EnergyFraction");
    produces<edm::ValueMap<float>>("chargedFromPV1EnergyFraction");
    produces<edm::ValueMap<float>>("chargedFromPV2EnergyFraction");
    produces<edm::ValueMap<float>>("chargedFromPV3EnergyFraction");
  }
  ~BetaStarVarProducer() override {};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  std::tuple<float,float,float,float> calculateCHSEnergies( edm::Ptr<pat::Jet> const & jet, double chefrompv0 ) const;

  edm::EDGetTokenT<edm::View<pat::Jet>> srcJet_;
  edm::EDGetTokenT<edm::View<T>> srcPF_;
  double maxDR_; 
};

template <typename T>
void
BetaStarVarProducer<T>::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
  edm::Handle<edm::View<pat::Jet>> srcJet;
  iEvent.getByToken(srcJet_, srcJet);
  edm::Handle<edm::View<T>> srcPF;
  iEvent.getByToken(srcPF_, srcPF);

  unsigned int nJet = srcJet->size();

  // Create an injective mapping from jets to charged PF candidates with fromPV==0 within the jet cone.
  // These are the PF candidates supposedly removed by the CHS method - i.e. removed charged pileup.
  std::vector< double > jet2pue(nJet,0);
  for ( unsigned int icand = 0; icand < srcPF->size(); ++icand ) {
    auto cand = srcPF->ptrAt(icand);
    if (cand->charge()==0 or cand->fromPV()!=0) continue;
    // Looking for the closest match for this charged frompv==0 candidate
    float dRMin = 999.;
    int bestjet = -1;
    for ( auto ijet=srcJet->begin(); ijet!=srcJet->end(); ++ijet ) {
      float dR = reco::deltaR(*ijet,*cand);
      if (dR<dRMin) {
        dRMin = dR;
        bestjet = ijet-srcJet->begin();
      }
    }
    // If the candidate is closer than the jet radius to the jet axis, this is a PU particle of interest for the selected jet
    if (dRMin<maxDR_) jet2pue[bestjet] += cand->energy();
  }

  std::vector<float> chargedFromPV0EnergyFraction(nJet,-1);
  std::vector<float> chargedFromPV1EnergyFraction(nJet,-1);
  std::vector<float> chargedFromPV2EnergyFraction(nJet,-1);
  std::vector<float> chargedFromPV3EnergyFraction(nJet,-1);

  for ( unsigned int ij = 0; ij < nJet; ++ij ) {
    auto ijet = srcJet->ptrAt(ij);
    auto vals = calculateCHSEnergies( ijet, jet2pue[ij] );
    auto chpuf0 = std::get<0>(vals);
    auto chpuf1 = std::get<1>(vals);
    auto chpuf2 = std::get<2>(vals);
    auto chpuf3 = std::get<3>(vals);
    //auto chef   = std::get<2>(vals);
    chargedFromPV0EnergyFraction[ij] = chpuf0;
    chargedFromPV1EnergyFraction[ij] = chpuf1;
    chargedFromPV2EnergyFraction[ij] = chpuf2;
    chargedFromPV3EnergyFraction[ij] = chpuf3;
  }

  std::unique_ptr<edm::ValueMap<float>> valuesFPV0(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerFPV0(*valuesFPV0);
  fillerFPV0.insert(srcJet,chargedFromPV0EnergyFraction.begin(),chargedFromPV0EnergyFraction.end());
  fillerFPV0.fill();
  iEvent.put(std::move(valuesFPV0),"chargedFromPV0EnergyFraction");

  std::unique_ptr<edm::ValueMap<float>> valuesFPV1(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerFPV1(*valuesFPV1);
  fillerFPV1.insert(srcJet,chargedFromPV1EnergyFraction.begin(),chargedFromPV1EnergyFraction.end());
  fillerFPV1.fill();
  iEvent.put(std::move(valuesFPV1),"chargedFromPV1EnergyFraction");

  std::unique_ptr<edm::ValueMap<float>> valuesFPV2(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerFPV2(*valuesFPV2);
  fillerFPV2.insert(srcJet,chargedFromPV2EnergyFraction.begin(),chargedFromPV2EnergyFraction.end());
  fillerFPV2.fill();
  iEvent.put(std::move(valuesFPV2),"chargedFromPV2EnergyFraction");

  std::unique_ptr<edm::ValueMap<float>> valuesFPV3(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerFPV3(*valuesFPV3);
  fillerFPV3.insert(srcJet,chargedFromPV3EnergyFraction.begin(),chargedFromPV3EnergyFraction.end());
  fillerFPV3.fill();
  iEvent.put(std::move(valuesFPV3),"chargedFromPV3EnergyFraction");
}

template <typename T>
std::tuple<float,float,float,float>
BetaStarVarProducer<T>::calculateCHSEnergies( edm::Ptr<pat::Jet> const & ijet, double chefrompv0 ) const {
  // Keep track of energy (present in the jet) for PU stuff
  double chefrompv1 = 0.0;
  double chefrompv2 = 0.0;
  double chefrompv3 = 0.0;

  // Loop through the PF candidates within the jet.
  // Store the sum of their energy, and their indices. 
  for (auto pidx = 0u; pidx < ijet->numberOfDaughters(); ++pidx) {
    auto *dtr = dynamic_cast<const pat::PackedCandidate*>(ijet->daughter(pidx));
    if (dtr->charge()==0) continue;
    if      (dtr->fromPV()==1) chefrompv1 += dtr->energy();
    else if (dtr->fromPV()==2) chefrompv2 += dtr->energy();
    else if (dtr->fromPV()==3) chefrompv3 += dtr->energy();
  }
  
  // Now get the fractions relative to the raw jet. 
  auto rawP4 = ijet->correctedP4(0);
  double chffpv0 = chefrompv0 / rawP4.energy();
  double chffpv1 = chefrompv1 / rawP4.energy(); 
  double chffpv2 = chefrompv2 / rawP4.energy(); 
  double chffpv3 = chefrompv3 / rawP4.energy(); 
  
  return std::tuple<float,float,float,float>(chffpv0,chffpv1,chffpv2,chffpv3);
}

template <typename T>
void
BetaStarVarProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcJet")->setComment("jet input collection");
  desc.add<edm::InputTag>("srcPF")->setComment("PF candidate input collection");
  desc.add<double>("maxDR")->setComment("Maximum DR to consider for jet-->pf cand association");
  std::string modname ("BetaStarVarProducer");
  descriptions.add(modname,desc);
}

typedef BetaStarVarProducer<pat::PackedCandidate> BetaStarPackedCandidateVarProducer;

DEFINE_FWK_MODULE(BetaStarPackedCandidateVarProducer);

