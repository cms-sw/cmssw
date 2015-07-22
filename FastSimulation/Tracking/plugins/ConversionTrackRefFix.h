#ifndef FastSimulation_Tracking_ConversionTrackRefFix_h
#define FastSimulation_Tracking_ConversionTrackRefFix_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaTrackReco/interface/ConversionTrackFwd.h"

class ConversionTrackRefFix : public edm::stream::EDProducer<>
{
 public:
  explicit ConversionTrackRefFix(const edm::ParameterSet&);
  ~ConversionTrackRefFix();
  
 private:
  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  edm::EDGetTokenT<reco::ConversionTrackCollection > conversionTracksToken;
  edm::EDGetTokenT<reco::TrackCollection > newTracksToken;
};

#endif
