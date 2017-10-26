// -*- C++ -*-
//
// Package:   RecoEgamma/HGCalElectronIDValueMapProducer
// Class:    HGCalElectronIDValueMapProducer
//
/**\class HGCalElectronIDValueMapProducer HGCalElectronIDValueMapProducer.cc RecoEgamma/HGCalElectronIDValueMapProducer/plugins/HGCalElectronIDValueMapProducer.cc

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

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include "RecoEgamma/EgammaTools/interface/ElectronIDHelper.h"
#include "RecoEgamma/EgammaTools/interface/LongDeps.h"

class HGCalElectronIDValueMapProducer : public edm::stream::EDProducer<> {
  public:
    explicit HGCalElectronIDValueMapProducer(const edm::ParameterSet&);
    ~HGCalElectronIDValueMapProducer() override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void beginStream(edm::StreamID) override;
    void produce(edm::Event&, const edm::EventSetup&) override;
    void endStream() override;

    // ----------member data ---------------------------
    edm::EDGetTokenT<edm::View<reco::GsfElectron>> ElectronsToken_;
    float radius_;
    std::map<const std::string, std::vector<float>> maps_;

    std::unique_ptr<ElectronIDHelper> eIDHelper_;
};

HGCalElectronIDValueMapProducer::HGCalElectronIDValueMapProducer(const edm::ParameterSet& iConfig) :
  ElectronsToken_(consumes<edm::View<reco::GsfElectron>>(iConfig.getParameter<edm::InputTag>("electrons"))),
  radius_(iConfig.getParameter<double>("pcaRadius"))
{
  // All the ValueMap names to output are defined in the python config
  // so that potential consumers can configure themselves in a simple manner
  for(auto key : iConfig.getParameter<std::vector<std::string>>("variables")) {
    maps_[key] = {};
    produces<edm::ValueMap<float>>(key);
  }

  eIDHelper_.reset(new ElectronIDHelper(iConfig, consumesCollector()));
}


HGCalElectronIDValueMapProducer::~HGCalElectronIDValueMapProducer()
{
}


// ------------ method called to produce the data  ------------
void
HGCalElectronIDValueMapProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  Handle<edm::View<reco::GsfElectron>> ElectronsH;
  iEvent.getByToken(ElectronsToken_, ElectronsH);

  const size_t prevMapSize = maps_.size();

  // Clear previous map
  for(auto&& kv : maps_) kv.second.clear();

  // Set up helper tool
  eIDHelper_->eventInit(iEvent,iSetup);

  for(size_t iEle=0; iEle<ElectronsH->size(); ++iEle) {
    const auto& electron = ElectronsH->at(iEle);

    if(electron.isEB()) {
      // Fill some dummy value
      for(auto&& kv : maps_) {
	kv.second.push_back(0.);
      }
    }
    else {
      eIDHelper_->computeHGCAL(electron, radius_);

      // check the PCA has worked out
      if (eIDHelper_->sigmaUU() == -1){
	  for(auto&& kv : maps_) {
	      kv.second.push_back(0.);
	  }
	  continue;
      }

      LongDeps ld(eIDHelper_->energyPerLayer(radius_, true));
      float measuredDepth, expectedDepth, expectedSigma;
      float depthCompatibility = eIDHelper_->clusterDepthCompatibility(ld, measuredDepth, expectedDepth, expectedSigma);

      // Fill here all the ValueMaps from their appropriate functions

      // Energies / PT
      maps_["gsfTrackPt"].push_back(electron.gsfTrack()->pt());
      maps_["pOutPt"].push_back(std::sqrt(electron.trackMomentumAtVtx().perp2()));
      maps_["scEt"].push_back(electron.superCluster()->energy() / std::cosh(electron.superCluster()->eta()));
      maps_["scEnergy"].push_back(electron.superCluster()->energy());
      maps_["ecOrigEt"].push_back(electron.electronCluster()->energy() / std::cosh(electron.electronCluster()->eta()));
      maps_["ecOrigEnergy"].push_back(electron.electronCluster()->energy());

      // energies calculated in an cylinder around the axis of the electron cluster
      float ec_tot_energy = ld.energyEE() + ld.energyFH() + ld.energyBH();
      maps_["ecEt"].push_back(ec_tot_energy / std::cosh(electron.electronCluster()->eta()));
      maps_["ecEnergy"].push_back(ec_tot_energy);
      maps_["ecEnergyEE"].push_back(ld.energyEE());
      maps_["ecEnergyFH"].push_back(ld.energyFH());
      maps_["ecEnergyBH"].push_back(ld.energyBH());

      // Track-based
      maps_["fbrem"].push_back( electron.fbrem() );
      maps_["gsfTrackHits"].push_back( electron.gsfTrack()->hitPattern().trackerLayersWithMeasurement() );
      maps_["gsfTrackChi2"].push_back( electron.gsfTrack()->normalizedChi2() );

      reco::TrackRef myTrackRef = electron.closestCtfTrackRef();
      bool validKF = myTrackRef.isAvailable() && myTrackRef.isNonnull();
      maps_["kfTrackHits"].push_back( (validKF) ? myTrackRef->hitPattern().trackerLayersWithMeasurement() : -1 );
      maps_["kfTrackChi2"].push_back( (validKF) ? myTrackRef->normalizedChi2() : -1 );

      // Track-matching
      maps_["dEtaTrackClust"].push_back( electron.deltaEtaEleClusterTrackAtCalo() );
      maps_["dPhiTrackClust"].push_back( electron.deltaPhiEleClusterTrackAtCalo() );

      // Cluster shapes
      // PCA related
      maps_["pcaEig1"].push_back(eIDHelper_->eigenValues()(0));
      maps_["pcaEig2"].push_back(eIDHelper_->eigenValues()(1));
      maps_["pcaEig3"].push_back(eIDHelper_->eigenValues()(2));
      maps_["pcaSig1"].push_back(eIDHelper_->sigmas()(0));
      maps_["pcaSig2"].push_back(eIDHelper_->sigmas()(1));
      maps_["pcaSig3"].push_back(eIDHelper_->sigmas()(2));

      // transverse shapes
      maps_["sigmaUU"].push_back(eIDHelper_->sigmaUU());
      maps_["sigmaVV"].push_back(eIDHelper_->sigmaVV());
      maps_["sigmaEE"].push_back(eIDHelper_->sigmaEE());
      maps_["sigmaPP"].push_back(eIDHelper_->sigmaPP());


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
      maps_["caloIsoRing0"].push_back(eIDHelper_->getIsolationRing(0));
      maps_["caloIsoRing1"].push_back(eIDHelper_->getIsolationRing(1));
      maps_["caloIsoRing2"].push_back(eIDHelper_->getIsolationRing(2));
      maps_["caloIsoRing3"].push_back(eIDHelper_->getIsolationRing(3));
      maps_["caloIsoRing4"].push_back(eIDHelper_->getIsolationRing(4));
    }
  }

  // Check we didn't make up a new variable and forget it in the constructor
  // (or some other pathology)
  if ( maps_.size() != prevMapSize ) {
    throw cms::Exception("HGCalElectronIDValueMapProducer") << "We have a miscoded value map producer, since map size changed";
  }

  for(auto&& kv : maps_) {
    // Check we didn't forget any values
    if ( kv.second.size() != ElectronsH->size() ) {
      throw cms::Exception("HGCalElectronIDValueMapProducer") << "We have a miscoded value map producer, since the variable " << kv.first << " wasn't filled.";
    }
    // Do the filling
    auto out = std::make_unique<edm::ValueMap<float>>();
    edm::ValueMap<float>::Filler filler(*out);
    filler.insert(ElectronsH, kv.second.begin(), kv.second.end());
    filler.fill();
    // and put it into the event
    iEvent.put(std::move(out), kv.first);
  }
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
HGCalElectronIDValueMapProducer::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
HGCalElectronIDValueMapProducer::endStream() {
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HGCalElectronIDValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalElectronIDValueMapProducer);
