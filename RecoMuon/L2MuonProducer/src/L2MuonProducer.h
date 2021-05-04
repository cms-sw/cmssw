#ifndef RecoMuon_L2MuonProducer_L2MuonProducer_H
#define RecoMuon_L2MuonProducer_L2MuonProducer_H

//-------------------------------------------------
//
/**  \class L2MuonProducer
 * 
 *   L2 muon reconstructor:
 *   reconstructs muons using DT, CSC and RPC
 *   information,<BR>
 *   starting from internal seeds (L2 muon track segments).
 *
 *
 *
 *   \author  R.Bellan - INFN TO
 *
 *   modified by A. Sharma to add fillDescription function
 */
//
//--------------------------------------------------

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
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

class L2MuonProducer : public edm::stream::EDProducer<> {
public:
  /// constructor with config
  L2MuonProducer(const edm::ParameterSet&);

  /// destructor
  ~L2MuonProducer() override;

  /// reconstruct muons
  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // MuonSeed Collection Label
  edm::InputTag theSeedCollectionLabel;

  /// the track finder
  std::unique_ptr<MuonTrackFinder> theTrackFinder;  //It isn't the same as in ORCA

  /// the event setup proxy, it takes care the services update
  std::unique_ptr<MuonServiceProxy> theService;

  edm::EDGetTokenT<edm::View<TrajectorySeed> > seedsToken;
};

#endif
