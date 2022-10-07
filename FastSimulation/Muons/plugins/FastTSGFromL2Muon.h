#ifndef FastSimulation_Muons_FastTSGFromL2Muon_H
#define FastSimulation_Muons_FastTSGFromL2Muon_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include <vector>
namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class MuonServiceProxy;
class MuonTrackingRegionBuilder;
class RectangularEtaPhiTrackingRegion;
class SimTrack;

//
// generate seeds corresponding to L2 muons
//

class FastTSGFromL2Muon : public edm::stream::EDProducer<> {
public:
  FastTSGFromL2Muon(const edm::ParameterSet& cfg);
  ~FastTSGFromL2Muon() override = default;
  void beginRun(edm::Run const& run, edm::EventSetup const& es) override;
  void produce(edm::Event& ev, const edm::EventSetup& es) override;

private:
  bool clean(reco::TrackRef muRef,
             RectangularEtaPhiTrackingRegion* region,
             const BasicTrajectorySeed* aSeed,
             const SimTrack& theSimTrack);

private:
  const double thePtCut;
  const edm::InputTag theL2CollectionLabel;
  const std::vector<edm::InputTag> theSeedCollectionLabels;
  const edm::InputTag theSimTrackCollectionLabel;

  const edm::EDGetTokenT<edm::SimTrackContainer> simTrackToken_;
  const edm::EDGetTokenT<reco::TrackCollection> l2TrackToken_;
  std::vector<edm::EDGetTokenT<edm::View<TrajectorySeed> > > seedToken_;

  MuonServiceProxy* theService;
  std::unique_ptr<MuonTrackingRegionBuilder> theRegionBuilder;
};
#endif
