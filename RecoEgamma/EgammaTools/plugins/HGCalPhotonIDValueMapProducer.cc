// -*- C++ -*-
//
// Package:   RecoEgamma/HGCalPhotonIDValueMapProducer
// Class:    HGCalPhotonIDValueMapProducer
//
/**\class HGCalPhotonIDValueMapProducer HGCalPhotonIDValueMapProducer.cc RecoEgamma/HGCalPhotonIDValueMapProducer/plugins/HGCalPhotonIDValueMapProducer.cc

 Description: [one line class summary]

 Implementation:
    [Notes on implementation]
*/
//
// Original Author:  Nicholas Charles Smith
//      Created:  Wed, 05 Apr 2017 12:17:43 GMT
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

#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "RecoEgamma/EgammaTools/interface/PhotonIDHelper.h"
#include "RecoEgamma/EgammaTools/interface/LongDeps.h"

class HGCalPhotonIDValueMapProducer : public edm::stream::EDProducer<> {
  public:
    explicit HGCalPhotonIDValueMapProducer(const edm::ParameterSet&);
    ~HGCalPhotonIDValueMapProducer() override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void beginStream(edm::StreamID) override;
    void produce(edm::Event&, const edm::EventSetup&) override;
    void endStream() override;

    // ----------member data ---------------------------
    edm::EDGetTokenT<edm::View<reco::Photon>> photonsToken_;
    float radius_;
    std::map<const std::string, std::vector<float>> maps_;

    std::unique_ptr<PhotonIDHelper> phoIDHelper_;
};

HGCalPhotonIDValueMapProducer::HGCalPhotonIDValueMapProducer(const edm::ParameterSet& iConfig) :
  photonsToken_(consumes<edm::View<reco::Photon>>(iConfig.getParameter<edm::InputTag>("photons"))),
  radius_(iConfig.getParameter<double>("pcaRadius"))
{
  // All the ValueMap names to output are defined in the python config
  // so that potential consumers can configure themselves in a simple manner
  for(auto key : iConfig.getParameter<std::vector<std::string>>("variables")) {
    maps_[key] = {};
    produces<edm::ValueMap<float>>(key);
  }

  phoIDHelper_.reset(new PhotonIDHelper(iConfig, consumesCollector()));
}


HGCalPhotonIDValueMapProducer::~HGCalPhotonIDValueMapProducer()
{
}


// ------------ method called to produce the data  ------------
void
HGCalPhotonIDValueMapProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  Handle<edm::View<reco::Photon>> photonsH;
  iEvent.getByToken(photonsToken_, photonsH);

  const size_t prevMapSize = maps_.size();

  // Clear previous map
  for(auto&& kv : maps_) kv.second.clear();

  // Set up helper tool
  phoIDHelper_->eventInit(iEvent,iSetup);

  for(size_t iPho=0; iPho<photonsH->size(); ++iPho) {
    const auto& pho = photonsH->at(iPho);

    if(pho.isEB()) {
      // Fill some dummy value
      for(auto&& kv : maps_) {
        kv.second.push_back(0.);
      }
    }
    else {
      phoIDHelper_->computeHGCAL(pho, radius_);

      // check the PCA has worked out
      if (phoIDHelper_->sigmaUU() == -1){
	  for(auto&& kv : maps_) {
	      kv.second.push_back(0.);
	  }
	  continue;
      }

      LongDeps ld(phoIDHelper_->energyPerLayer(radius_, true));
      float measuredDepth, expectedDepth, expectedSigma;
      float depthCompatibility = phoIDHelper_->clusterDepthCompatibility(ld, measuredDepth, expectedDepth, expectedSigma);

      // Fill here all the ValueMaps from their appropriate functions

      // Energies / PT
      maps_["scEt"].push_back(pho.superCluster()->energy() / std::cosh(pho.superCluster()->eta()));
      maps_["scEnergy"].push_back(pho.superCluster()->energy());
      maps_["seedOrigEt"].push_back(pho.superCluster()->seed()->energy() / std::cosh(pho.superCluster()->seed()->eta()));
      maps_["seedOrigEnergy"].push_back(pho.superCluster()->seed()->energy());

      // energies calculated in an cylinder around the axis of the pho cluster
      float seed_tot_energy = ld.energyEE() + ld.energyFH() + ld.energyBH();
      maps_["seedEt"].push_back(seed_tot_energy / std::cosh(pho.superCluster()->seed()->eta()));
      maps_["seedEnergy"].push_back(seed_tot_energy);
      maps_["seedEnergyEE"].push_back(ld.energyEE());
      maps_["seedEnergyFH"].push_back(ld.energyFH());
      maps_["seedEnergyBH"].push_back(ld.energyBH());

      // Cluster shapes
      // PCA related
      maps_["pcaEig1"].push_back(phoIDHelper_->eigenValues()(0));
      maps_["pcaEig2"].push_back(phoIDHelper_->eigenValues()(1));
      maps_["pcaEig3"].push_back(phoIDHelper_->eigenValues()(2));
      maps_["pcaSig1"].push_back(phoIDHelper_->sigmas()(0));
      maps_["pcaSig2"].push_back(phoIDHelper_->sigmas()(1));
      maps_["pcaSig3"].push_back(phoIDHelper_->sigmas()(2));

      // transverse shapes
      maps_["sigmaUU"].push_back(phoIDHelper_->sigmaUU());
      maps_["sigmaVV"].push_back(phoIDHelper_->sigmaVV());
      maps_["sigmaEE"].push_back(phoIDHelper_->sigmaEE());
      maps_["sigmaPP"].push_back(phoIDHelper_->sigmaPP());


      // long profile
      maps_["nLayers"].push_back(ld.nLayers());
      maps_["firstLayer"].push_back(ld.firstLayer());
      maps_["lastLayer"].push_back(ld.lastLayer());
      maps_["e4oEtot"].push_back(ld.e4oEtot());
      maps_["layerEfrac10"].push_back(ld.layerEfrac10());
      maps_["layerEfrac90"].push_back(ld.layerEfrac90());
      //maps_["firstLayerEnergy"].push_back(ld.energyPerLayer()[ld.firstLayer()]);

      // depth
      maps_["measuredDepth"].push_back(measuredDepth);
      maps_["expectedDepth"].push_back(expectedDepth);
      maps_["expectedSigma"].push_back(expectedSigma);
      maps_["depthCompatibility"].push_back(depthCompatibility);

      // Isolation
      maps_["caloIsoRing0"].push_back(phoIDHelper_->getIsolationRing(0));
      maps_["caloIsoRing1"].push_back(phoIDHelper_->getIsolationRing(1));
      maps_["caloIsoRing2"].push_back(phoIDHelper_->getIsolationRing(2));
      maps_["caloIsoRing3"].push_back(phoIDHelper_->getIsolationRing(3));
      maps_["caloIsoRing4"].push_back(phoIDHelper_->getIsolationRing(4));
    }
  }

  // Check we didn't make up a new variable and forget it in the constructor
  // (or some other pathology)
  if ( maps_.size() != prevMapSize ) {
    throw cms::Exception("HGCalPhotonIDValueMapProducer") << "We have a miscoded value map producer, since map size changed";
  }

  for(auto&& kv : maps_) {
    // Check we didn't forget any values
    if ( kv.second.size() != photonsH->size() ) {
      throw cms::Exception("HGCalPhotonIDValueMapProducer") << "We have a miscoded value map producer, since the variable " << kv.first << " wasn't filled.";
    }
    // Do the filling
    auto out = std::make_unique<edm::ValueMap<float>>();
    edm::ValueMap<float>::Filler filler(*out);
    filler.insert(photonsH, kv.second.begin(), kv.second.end());
    filler.fill();
    // and put it into the event
    iEvent.put(std::move(out), kv.first);
  }
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
HGCalPhotonIDValueMapProducer::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
HGCalPhotonIDValueMapProducer::endStream() {
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HGCalPhotonIDValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalPhotonIDValueMapProducer);
