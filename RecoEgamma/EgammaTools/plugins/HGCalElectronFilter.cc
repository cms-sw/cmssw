// -*- C++ -*-
//
// Package:    EgammaTools/HGCalElectronFilter
// Class:      HGCalElectronFilter
//
/**\class HGCalElectronFilter HGCalElectronFilter.cc EgammaTools/HGCalElectronFilter/plugins/HGCalElectronFilter.cc

 Description: filtering of duplicate HGCAL electrons

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Florian beaudette
//         Created:  Mon, 06 Nov 2017 21:49:54 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

//
// class declaration
//

class HGCalElectronFilter : public edm::stream::EDProducer<> {
public:
  explicit HGCalElectronFilter(const edm::ParameterSet&);
  ~HGCalElectronFilter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<edm::View<reco::GsfElectron>> electronsToken_;
  std::string outputCollection_;
  bool cleanBarrel_;
};

HGCalElectronFilter::HGCalElectronFilter(const edm::ParameterSet& iConfig)
    : electronsToken_(consumes(iConfig.getParameter<edm::InputTag>("inputGsfElectrons"))),
      outputCollection_(iConfig.getParameter<std::string>("outputCollection")),
      cleanBarrel_(iConfig.getParameter<bool>("cleanBarrel")) {
  produces<reco::GsfElectronCollection>(outputCollection_);
}

HGCalElectronFilter::~HGCalElectronFilter() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void HGCalElectronFilter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto gsfElectrons_p = std::make_unique<reco::GsfElectronCollection>();

  auto electronsH = iEvent.getHandle(electronsToken_);

  for (const auto& electron1 : *electronsH) {
    bool isBest = true;
    if (!cleanBarrel_ && electron1.isEB()) {  // keep all barrel electrons
      gsfElectrons_p->push_back(electron1);
      continue;
    } else {
      for (const auto& electron2 : *electronsH) {
        if (&electron1 == &electron2)
          continue;
        if (electron1.superCluster() != electron2.superCluster())
          continue;
        if (electron1.electronCluster() != electron2.electronCluster()) {
          if (electron1.electronCluster()->energy() < electron2.electronCluster()->energy()) {
            isBest = false;
            break;
          }
        } else {
          if (fabs(electron1.eEleClusterOverPout() - 1.) > fabs(electron2.eEleClusterOverPout() - 1.)) {
            isBest = false;
            break;
          }
        }
      }
      if (isBest)
        gsfElectrons_p->push_back(electron1);
    }
  }

  iEvent.put(std::move(gsfElectrons_p), outputCollection_);
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void HGCalElectronFilter::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void HGCalElectronFilter::endStream() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HGCalElectronFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // cleanedEcalDrivenGsfElectronsFromMultiCl
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputGsfElectrons", edm::InputTag("ecalDrivenGsfElectronsFromMultiCl"));
  desc.add<bool>("cleanBarrel", false);
  desc.add<std::string>("outputCollection", "");
  descriptions.add("cleanedEcalDrivenGsfElectronsFromMultiCl", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalElectronFilter);
