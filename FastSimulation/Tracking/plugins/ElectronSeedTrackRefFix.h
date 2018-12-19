// system include files
#ifndef FastSimulation_Tracking_ElectronSeedTrackRefFix_h
#define FastSimulation_Tracking_ElectronSeedTrackRefFix_h


#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/ParticleFlowReco/interface/PreIdFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/Common/interface/ValueMap.h"


class ElectronSeedTrackRefFix : public edm::stream::EDProducer<> {
public:
  explicit ElectronSeedTrackRefFix(const edm::ParameterSet&);
  ~ElectronSeedTrackRefFix() override;
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  void produce(edm::Event&, const edm::EventSetup&) override;
    
  // ----------member data ---------------------------
  edm::EDGetTokenT<reco::TrackCollection > newTracksToken_;
  edm::EDGetTokenT<reco::TrackCollection > oldTracksToken_;
  edm::EDGetTokenT<reco::ElectronSeedCollection > seedsToken_;
  std::vector<edm::EDGetTokenT<reco::PreIdCollection > > idsToken_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<reco::PreIdRef>  > > idMapToken_;

  std::string preidgsfLabel_;
  std::vector<std::string> preidLabel_;
  edm::InputTag oldTracksTag_;
  edm::InputTag newTracksTag_;
  edm::InputTag seedsTag_;
  std::vector<edm::InputTag> idsTag_;
  
};

#endif
