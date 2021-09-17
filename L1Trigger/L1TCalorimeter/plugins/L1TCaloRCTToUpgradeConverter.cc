// -*- C++ -*-
//
// Package:    L1Trigger/L1TCalorimeter
// Class:      L1TCaloRCTToUpgradeConverter
//
/**\class l1t::L1TCaloRCTToUpgradeConverter L1TCaloRCTToUpgradeConverter.cc L1Trigger/L1TCalorimeter/plugins/L1TCaloRCTToUpgradeConverter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  James Brooke
//         Created:  Thu, 05 Dec 2013 17:39:27 GMT
//
//
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"
#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include <vector>

namespace l1t {

  class L1TCaloRCTToUpgradeConverter : public edm::global::EDProducer<> {
  public:
    explicit L1TCaloRCTToUpgradeConverter(const edm::ParameterSet& ps);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

    // ----------member data ---------------------------

    edm::EDGetTokenT<L1CaloRegionCollection> const rgnToken_;
    edm::EDGetTokenT<L1CaloEmCollection> const emToken_;

    edm::EDPutTokenT<CaloRegionBxCollection> const rgnPutToken_;
    edm::EDPutTokenT<CaloEmCandBxCollection> const emPutToken_;
  };

}  // namespace l1t

using namespace l1t;

L1TCaloRCTToUpgradeConverter::L1TCaloRCTToUpgradeConverter(const edm::ParameterSet& ps)
    : rgnToken_{consumes<L1CaloRegionCollection>(ps.getParameter<edm::InputTag>("regionTag"))},
      emToken_{consumes<L1CaloEmCollection>(ps.getParameter<edm::InputTag>("emTag"))},
      rgnPutToken_{produces<CaloRegionBxCollection>()},
      emPutToken_{produces<CaloEmCandBxCollection>()} {}

// ------------ method called to produce the data  ------------
void L1TCaloRCTToUpgradeConverter::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // check status of RCT conditions & renew if needed

  // store new formats
  BXVector<CaloEmCand> emcands;
  BXVector<CaloRegion> regions;

  // get old formats
  auto const& ems = iEvent.get(emToken_);
  auto const& rgns = iEvent.get(rgnToken_);

  // get the firstBx_ and lastBx_ from the input datatypes (assume bx for em same as rgn)
  int firstBx = 0;
  int lastBx = 0;
  for (auto const& em : ems) {
    int bx = em.bx();
    if (bx < firstBx)
      firstBx = bx;
    if (bx > lastBx)
      lastBx = bx;
  }

  emcands.setBXRange(firstBx, lastBx);
  regions.setBXRange(firstBx, lastBx);

  const ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > p4(0, 0, 0, 0);

  // loop over EM
  for (auto const& em : ems) {
    // get physical units
    // double pt = 0.;
    // double eta = 0.;
    // double phi = 0.;
    //math::PtEtaPhiMLorentzVector p4( pt+1.e-6, eta, phi, 0. );

    //CaloStage1Cluster cluster;
    CaloEmCand EmCand(p4, (int)em.rank(), (int)em.regionId().ieta(), (int)em.regionId().iphi(), (int)em.index());

    EmCand.setHwIso((int)em.isolated());
    //std::cout<<"ISO:    "<<EmCand.hwIso()<<"    "<<em.isolated()<<std::endl;

    // create new format
    emcands.push_back(em.bx(), EmCand);
  }

  // loop over regions
  for (auto const& rgn : rgns) {
    // get physical units
    // double pt = 0.;
    // double eta = 0.;
    // double phi = 0.;
    //math::PtEtaPhiMLorentzVector p4( pt+1.e-6, eta, phi, 0 );

    bool tauVeto = rgn.fineGrain();  //equivalent to tauVeto for HB/HE, includes extra info for HF
    int hwQual = (int)tauVeto;

    // create new format
    // several values here are stage 2 only, leave empty
    CaloRegion region(p4,                    //  LorentzVector& p4,
                      0.,                    //  etEm,
                      0.,                    //  etHad,
                      (int)rgn.et(),         //  pt,
                      (int)rgn.id().ieta(),  //  eta,
                      (int)rgn.id().iphi(),  //  phi,
                      hwQual,                //  qual,
                      0,                     //  hwEtEm,
                      0);                    //  hwEtHad

    // add to output
    regions.push_back(rgn.bx(), region);
  }

  iEvent.emplace(emPutToken_, std::move(emcands));
  iEvent.emplace(rgnPutToken_, std::move(regions));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1TCaloRCTToUpgradeConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("regionTag");
  desc.add<edm::InputTag>("emTag");
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TCaloRCTToUpgradeConverter);
