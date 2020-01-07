#ifndef RecoMuon_L3MuonProducer_L3MuonProducer_H
#define RecoMuon_L3MuonProducer_L3MuonProducer_H

/**  \class L3MuonProducer
 * 
 *   L3 muon reconstructor:
 *   reconstructs muons using DT, CSC, RPC and tracker
 *   information,<BR>
 *   starting from a L2 reonstructed muon.
 *
 *   \author  A. Everett - Purdue University
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"

// Input and output collection
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include <memory>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class MuonTrackFinder;
class MuonServiceProxy;

class L3MuonProducer : public edm::stream::EDProducer<> {
public:
  /// constructor with config
  L3MuonProducer(const edm::ParameterSet&);

  /// destructor
  ~L3MuonProducer() override;

  /// reconstruct muons
  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  /// Seed STA Label
  edm::InputTag theL2CollectionLabel;

  /// Label for L2SeededTracks
  std::string theL2SeededTkLabel;

  edm::EDGetTokenT<reco::TrackCollection> l2MuonToken_;
  edm::EDGetTokenT<std::vector<Trajectory> > l2MuonTrajToken_;
  edm::EDGetTokenT<TrajTrackAssociationCollection> l2AssoMapToken_;
  edm::EDGetTokenT<reco::TrackToTrackMap> updatedL2AssoMapToken_;

  std::unique_ptr<MuonTrackFinder> theTrackFinder;

  /// the event setup proxy, it takes care the services update
  std::unique_ptr<MuonServiceProxy> theService;
};

#endif
