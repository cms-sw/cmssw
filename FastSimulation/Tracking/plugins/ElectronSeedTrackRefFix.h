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
  ~ElectronSeedTrackRefFix();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
    
  // ----------member data ---------------------------
  edm::EDGetTokenT<reco::TrackCollection > newTracksToken;
  edm::EDGetTokenT<reco::TrackCollection > oldTracksToken;
  edm::EDGetTokenT<reco::ElectronSeedCollection > seedsToken;
  edm::EDGetTokenT<reco::PreIdCollection > idsToken;
  edm::EDGetTokenT<edm::ValueMap<reco::PreIdRef>  > idMapToken;

  std::string preidgsfLabel;
  std::string preidLabel;
  edm::InputTag oldTracksTag;
  edm::InputTag newTracksTag;
  edm::InputTag seedsTag;
  edm::InputTag idsTag;
  
};

#endif
