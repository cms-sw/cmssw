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
    produces<edm::ValueMap<float>>("chargedHadronPUEnergyFraction");
    produces<edm::ValueMap<float>>("chargedHadronCHSEnergyFraction");
  }
  ~BetaStarVarProducer() override {};

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  std::tuple<float,float> calculateCHSEnergies( edm::Ptr<pat::Jet> const & jet, edm::View<T> const & cands) const;

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

  std::vector<float> chargedHadronPUEnergyFraction(nJet,-1);
  std::vector<float> chargedHadronCHSEnergyFraction(nJet,-1);

  for ( unsigned int ij = 0; ij < nJet; ++ij ) {
    auto jet = srcJet->ptrAt(ij);
    std::tuple<float,float> vals = calculateCHSEnergies( jet, *srcPF );
    auto chpuf = std::get<0>(vals);
    auto chef  = std::get<1>(vals);
    chargedHadronPUEnergyFraction[ij] = chpuf;
    chargedHadronCHSEnergyFraction[ij] = chef;
  }

  std::unique_ptr<edm::ValueMap<float>> chargedHadronPUEnergyFractionV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerPU(*chargedHadronPUEnergyFractionV);
  fillerPU.insert(srcJet,chargedHadronPUEnergyFraction.begin(),chargedHadronPUEnergyFraction.end());
  fillerPU.fill();
  iEvent.put(std::move(chargedHadronPUEnergyFractionV),"chargedHadronPUEnergyFraction");
  
  std::unique_ptr<edm::ValueMap<float>> chargedHadronCHSEnergyFractionV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerCHE(*chargedHadronCHSEnergyFractionV);
  fillerCHE.insert(srcJet,chargedHadronCHSEnergyFraction.begin(),chargedHadronCHSEnergyFraction.end());
  fillerCHE.fill();
  iEvent.put(std::move(chargedHadronCHSEnergyFractionV),"chargedHadronCHSEnergyFraction");

}

template <typename T>
std::tuple<float,float>
BetaStarVarProducer<T>::calculateCHSEnergies( edm::Ptr<pat::Jet> const & ijet, edm::View<T> const & cands ) const {

  auto rawP4 = ijet->correctedP4(0);
  std::vector<unsigned int> jet2pu;

  // Get all of the PF candidates within a cone of dR to the jet that
  // do NOT belong to the primary vertex.
  // Store their indices. 
  for ( unsigned int icand = 0; icand < cands.size(); ++icand ) {
    auto cand = cands.ptrAt(icand);
    if (cand->fromPV()!=0) continue;
    float dR = reco::deltaR(*ijet,*cand);
    if (dR<maxDR_) {
      jet2pu.emplace_back( cand.key() );
    }
  }
  
  // Keep track of energy for PU stuff
  double che = 0.0;
  double pue = 0.0; 

  // Loop through the PF candidates within the jet.
  // Store the sum of their energy, and their indices. 
  std::vector<unsigned int> used;
  auto jetConstituents = ijet->daughterPtrVector();
  for (auto const & ic : jetConstituents ) {
    auto icpc = dynamic_cast<pat::PackedCandidate const *>(ic.get());
    if ( icpc->charge()!=0) {
      che += icpc->energy();
      if (icpc->fromPV()==0) {
	used.push_back( ic.key() );
      }
    }
  }
  // Loop through the pileup PF candidates within the jet.
  for (auto pidx : jet2pu) {
    auto const & dtr = cands.ptrAt(pidx);
    // We take the candidates that have not appeared before: these were removed by CHS
    if (dtr->charge()!=0 and std::find(used.begin(),used.end(),dtr.key() )==used.end())
      pue += dtr->energy();
  }
  
  // Now get the fractions relative to the raw jet. 
  auto puf = pue / rawP4.energy();
  auto chf = che / rawP4.energy();
  
  return std::tuple<float,float>(puf,chf);
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

