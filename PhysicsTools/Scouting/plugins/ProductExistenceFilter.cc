#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>

template <typename T>
class ProductExistenceFilter : public edm::global::EDFilter<> {
public:
  ProductExistenceFilter(const edm::ParameterSet &);
  ~ProductExistenceFilter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  bool filter(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::EDGetTokenT<T> productToken_;
};

template <typename T>
ProductExistenceFilter<T>::ProductExistenceFilter(const edm::ParameterSet &iConfig)
    : productToken_(consumes(iConfig.getParameter<edm::InputTag>("product"))) {}

template <typename T>
void ProductExistenceFilter<T>::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("product");
  descriptions.addWithDefaultLabel(desc);
}

template <typename T>
bool ProductExistenceFilter<T>::filter(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  return iEvent.getHandle(productToken_).isValid();
}

#include "DataFormats/Scouting/interface/Run3ScoutingMuon.h"
using Run3ScoutingMuonExistenceFilter = ProductExistenceFilter<Run3ScoutingMuonCollection>;
DEFINE_FWK_MODULE(Run3ScoutingMuonExistenceFilter);

#include "DataFormats/Scouting/interface/Run3ScoutingElectron.h"
using Run3ScoutingElectronExistenceFilter = ProductExistenceFilter<Run3ScoutingElectronCollection>;
DEFINE_FWK_MODULE(Run3ScoutingElectronExistenceFilter);

#include "DataFormats/Scouting/interface/Run3ScoutingPhoton.h"
using Run3ScoutingPhotonExistenceFilter = ProductExistenceFilter<Run3ScoutingPhotonCollection>;
DEFINE_FWK_MODULE(Run3ScoutingPhotonExistenceFilter);

#include "DataFormats/Scouting/interface/Run3ScoutingTrack.h"
using Run3ScoutingTrackExistenceFilter = ProductExistenceFilter<Run3ScoutingTrackCollection>;
DEFINE_FWK_MODULE(Run3ScoutingTrackExistenceFilter);

#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"
using Run3ScoutingVertexExistenceFilter = ProductExistenceFilter<Run3ScoutingVertexCollection>;
DEFINE_FWK_MODULE(Run3ScoutingVertexExistenceFilter);

#include "DataFormats/Scouting/interface/Run3ScoutingParticle.h"
using Run3ScoutingParticleExistenceFilter = ProductExistenceFilter<Run3ScoutingParticleCollection>;
DEFINE_FWK_MODULE(Run3ScoutingParticleExistenceFilter);

#include "DataFormats/Scouting/interface/Run3ScoutingPFJet.h"
using Run3ScoutingPFJetExistenceFilter = ProductExistenceFilter<Run3ScoutingPFJetCollection>;
DEFINE_FWK_MODULE(Run3ScoutingPFJetExistenceFilter);

// MET and Rho
using DoubleExistenceFilter = ProductExistenceFilter<double>;
DEFINE_FWK_MODULE(DoubleExistenceFilter);
