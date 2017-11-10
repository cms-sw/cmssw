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
      ~HGCalElectronFilter();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginStream(edm::StreamID) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endStream() override;

      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------
      edm::EDGetTokenT<edm::View<reco::GsfElectron>> ElectronsToken_;
      std::string  outputCollection_;
      bool cleanBarrel_;
};


HGCalElectronFilter::HGCalElectronFilter(const edm::ParameterSet& iConfig):
    ElectronsToken_(consumes<edm::View<reco::GsfElectron>>(iConfig.getParameter<edm::InputTag>("inputGsfElectrons"))),
    outputCollection_(iConfig.getParameter<std::string>("outputCollection")),
    cleanBarrel_(iConfig.getParameter<bool>("cleanBarrel"))
{
    produces<reco::GsfElectronCollection>(outputCollection_);
}


HGCalElectronFilter::~HGCalElectronFilter() {
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
HGCalElectronFilter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    auto gsfElectrons_p = std::make_unique<reco::GsfElectronCollection>();

    edm::Handle<edm::View<reco::GsfElectron>> ElectronsH;
    iEvent.getByToken(ElectronsToken_, ElectronsH);

    unsigned nElectrons = ElectronsH->size();

    if (nElectrons > 0 ) {
        for(unsigned iEle1 = 0; iEle1 < nElectrons; ++iEle1) {
            bool isBest = true;
            const auto& electron1 = ElectronsH->at(iEle1);
            if (!cleanBarrel_ && electron1.isEB()) {// keep all barrel electrons
                isBest = true;
                gsfElectrons_p->push_back(electron1);
                continue;
            } else {
                for (unsigned iEle2 = 0; iEle2 < nElectrons; ++iEle2) {
                    if (iEle1 == iEle2) continue;
                    const auto& electron2 = ElectronsH->at(iEle2);
                    if (electron1.superCluster() != electron2.superCluster()) continue;
                    if (electron1.electronCluster() != electron2.electronCluster()) {
                        if (electron1.electronCluster()->energy() < electron2.electronCluster()->energy()) {
                            isBest=false;
                            break;
                        }
                    } else {
                        if (fabs(electron1.eEleClusterOverPout()-1.) > fabs(electron2.eEleClusterOverPout()-1.)) {
                            isBest=false;
                            break;
                        }
                    }
                }
                if (isBest) gsfElectrons_p->push_back(electron1);
            }
        }
    }

    iEvent.put(std::move(gsfElectrons_p),outputCollection_);
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
HGCalElectronFilter::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
HGCalElectronFilter::endStream() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
HGCalElectronFilter::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
HGCalElectronFilter::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
HGCalElectronFilter::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
HGCalElectronFilter::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HGCalElectronFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalElectronFilter);
