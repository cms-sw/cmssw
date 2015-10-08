// -*- C++ -*-
//
// Package:    L1TMicroGMTInputProducerFromGen
// Class:      L1TMicroGMTInputProducerFromGen
//
/**\class L1TMicroGMTInputProducerFromGen L1TMicroGMTInputProducerFromGen.cc L1Trigger/L1TGlobalMuon/plugins/L1TMicroGMTInputProducerFromGen.cc

 Description: takes generated muons and fills them in the expected collections for the MicroGMT

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Joschka Philip Lingemann,40 3-B01,+41227671598,
//         Created:  Thu Oct  3 10:12:30 CEST 2013
// $Id$
//
//


// system include files
#include <memory>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/GMTInputCaloSum.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "TMath.h"
#include "TRandom3.h"

//
// class declaration
//
using namespace l1t;

class L1TMicroGMTInputProducerFromGen : public edm::EDProducer {
   public:
      explicit L1TMicroGMTInputProducerFromGen(const edm::ParameterSet&);
      ~L1TMicroGMTInputProducerFromGen();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      virtual void beginRun(edm::Run&, edm::EventSetup const&);
      virtual void endRun(edm::Run&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

      static bool compareMuons(const RegionalMuonCand&, const RegionalMuonCand&);

      // ----------member data ---------------------------
      edm::EDGetTokenT <reco::GenParticleCollection> genParticlesToken;
      int m_currEvt;
      const static int m_maxMuons = 108;
      TRandom3 m_rnd;
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
L1TMicroGMTInputProducerFromGen::L1TMicroGMTInputProducerFromGen(const edm::ParameterSet& iConfig) :
  m_currEvt(0), m_rnd(0)
{
  //register your inputs:
  genParticlesToken = consumes <reco::GenParticleCollection> (std::string("genParticles"));
  //register your products
  produces<RegionalMuonCandBxCollection>("BarrelTFMuons");
  produces<RegionalMuonCandBxCollection>("OverlapTFMuons");
  produces<RegionalMuonCandBxCollection>("ForwardTFMuons");
  produces<GMTInputCaloSumBxCollection>("TriggerTowerSums");
}


L1TMicroGMTInputProducerFromGen::~L1TMicroGMTInputProducerFromGen()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

bool
L1TMicroGMTInputProducerFromGen::compareMuons(const RegionalMuonCand& mu1, const RegionalMuonCand& mu2)
{
  return mu1.processor() < mu2.processor();
}

// ------------ method called to produce the data  ------------
void
L1TMicroGMTInputProducerFromGen::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  std::auto_ptr<RegionalMuonCandBxCollection> barrelMuons (new RegionalMuonCandBxCollection());
  std::auto_ptr<RegionalMuonCandBxCollection> overlapMuons (new RegionalMuonCandBxCollection());
  std::auto_ptr<RegionalMuonCandBxCollection> endcapMuons (new RegionalMuonCandBxCollection());
  std::auto_ptr<GMTInputCaloSumBxCollection> towerSums (new GMTInputCaloSumBxCollection());

  std::vector<RegionalMuonCand> bmMuons;
  std::vector<RegionalMuonCand> omMuons;
  std::vector<RegionalMuonCand> emMuons;


  std::vector<int> muIndices;
  edm::Handle<reco::GenParticleCollection> genParticles;
  // Make sure that you can get genParticles
  if (iEvent.getByToken(genParticlesToken, genParticles)) {
    int cntr = 0;
    for (auto it = genParticles->cbegin(); it != genParticles->cend(); ++it) {
      const reco::Candidate& mcParticle = *it;
      if( abs(mcParticle.pdgId()) == 13 && mcParticle.status() == 1 )  muIndices.push_back(cntr);
      cntr++;
    }
  }
  else {
    LogTrace("GlobalMuon") << " GenParticleCollection not found." << std::endl;
  }

  RegionalMuonCand mu;
  GMTInputCaloSum tSum;
  // alternative scale (using full phi bit-width): 163.4521265553765f;
  const float phiToInt = 91.67324722093171f;
  // alternative scale: 100.0f;
  const float etaToInt = 90.9090909090f;
  const int maxPt = (1 << 9)-1;
  int muCntr = 0;

  double twoPi = TMath::Pi() * 2.;

  for (auto it = muIndices.begin(); it != muIndices.end(); ++it) {
    // don't really care which muons are taken...
    // guess there ain't 108 generated anyways
    if (muCntr == m_maxMuons) break;
    int gen_idx = *it;
    const reco::Candidate& mcMuon = genParticles->at(gen_idx);
    double eta = mcMuon.eta();
    if (fabs(eta) > 2.45) continue; // out of acceptance
    int hwPt = int(mcMuon.pt() * 2);
    hwPt = (hwPt < maxPt ? hwPt : maxPt);
    int hwEta = int(eta * etaToInt);
    double phi = mcMuon.phi();
    if (phi < 0) phi += twoPi; // add 2*pi
    int hwPhi = (int(phi * phiToInt))%576;
    int hwQual = 8;
    int hwCharge = (mcMuon.charge() > 0) ? 0 : 1;
    int hwChargeValid = 1;

    mu.setHwPt(hwPt);



    tftype tf(tftype::bmtf);
    int globalWedgePhi = (hwPhi+24)%576; // this sets CMS phi = 0 to -15 deg
    int localPhi = globalWedgePhi%48;
    int processor = globalWedgePhi / 48 + 1;
    int globalSectorPhi = (hwPhi-24); // this sets CMS phi = 0 to +15 deg
    if (globalSectorPhi < 0) {
      globalSectorPhi += 576;
    }


    if (fabs(eta) > 0.8) {
      if (fabs(eta) < 1.2) {
        tf = (eta > 0 ? tftype::omtf_pos : tftype::omtf_neg);
        processor = globalSectorPhi / 96 + 1;
        localPhi = globalSectorPhi%96;
      } else {
        tf = (eta > 0 ? tftype::emtf_pos : tftype::emtf_neg);
        processor = globalSectorPhi / 96 + 1;
        localPhi = globalSectorPhi%96;
      }
    }
    mu.setHwPhi(localPhi);
    mu.setTFIdentifiers(processor, tf);

    mu.setHwEta(hwEta);
    mu.setHwSign(hwCharge);
    mu.setHwSignValid(hwChargeValid);
    mu.setHwQual(hwQual);

    if (fabs(eta) < 0.8 && bmMuons.size() < 36) {
      bmMuons.push_back(mu);
      muCntr++;
    } else if (fabs(eta) < 1.2  && omMuons.size() < 36) {
      omMuons.push_back(mu);
      muCntr++;
    } else if (emMuons.size() < 36) {
      emMuons.push_back(mu);
      muCntr++;
    }
  }

  std::sort(bmMuons.begin(), bmMuons.end(), L1TMicroGMTInputProducerFromGen::compareMuons);
  std::sort(omMuons.begin(), omMuons.end(), L1TMicroGMTInputProducerFromGen::compareMuons);
  std::sort(emMuons.begin(), emMuons.end(), L1TMicroGMTInputProducerFromGen::compareMuons);

  for (const auto& mu:bmMuons) {
    barrelMuons->push_back(0, mu);
  }

  for (const auto& mu:omMuons) {
    overlapMuons->push_back(0, mu);
  }

  for (const auto& mu:emMuons) {
    endcapMuons->push_back(0, mu);
  }

  for (int i = 0; i < 1008; ++i) {
    // from where could I take the tower energies?
    int energy = int(m_rnd.Gaus(12, 6));
    if (energy < 0) energy = 0;
    if (energy > 31) energy = 31;
    GMTInputCaloSum sum(energy, i/28, i%28, i);
    towerSums->push_back(0, sum);
  }

  iEvent.put(barrelMuons, "BarrelTFMuons");
  iEvent.put(overlapMuons, "OverlapTFMuons");
  iEvent.put(endcapMuons, "ForwardTFMuons");
  iEvent.put(towerSums, "TriggerTowerSums");
  m_currEvt++;

}

// ------------ method called once each job just before starting event loop  ------------
void
L1TMicroGMTInputProducerFromGen::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1TMicroGMTInputProducerFromGen::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void
L1TMicroGMTInputProducerFromGen::beginRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void
L1TMicroGMTInputProducerFromGen::endRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void
L1TMicroGMTInputProducerFromGen::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void
L1TMicroGMTInputProducerFromGen::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TMicroGMTInputProducerFromGen::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TMicroGMTInputProducerFromGen);
