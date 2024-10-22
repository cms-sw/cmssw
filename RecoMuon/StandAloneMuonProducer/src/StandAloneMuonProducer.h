#ifndef RecoMuon_StandAloneMuonProducer_StandAloneMuonProducer_H
#define RecoMuon_StandAloneMuonProducer_StandAloneMuonProducer_H

/**  \class StandAloneMuonProducer
 * 
 *   StandAlone muon reconstructor:
 *   reconstructs muons using DT, CSC and RPC
 *   information,<BR>
 *   starting from internal seeds (L2 muon track segments).
 *
 *
 *
 *   \author  R.Bellan - INFN TO
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include <memory>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class MuonTrackFinder;
class MuonServiceProxy;

class StandAloneMuonProducer : public edm::stream::EDProducer<> {
public:
  /// constructor with config
  StandAloneMuonProducer(const edm::ParameterSet&);

  /// destructor
  ~StandAloneMuonProducer() override;

  /// reconstruct muons
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  /// MuonSeed Collection Label
  edm::InputTag theSeedCollectionLabel;

  /// the track finder
  std::unique_ptr<MuonTrackFinder> theTrackFinder;  //It isn't the same as in ORCA

  /// the event setup proxy, it takes care the services update
  std::unique_ptr<MuonServiceProxy> theService;

  edm::EDGetTokenT<edm::View<TrajectorySeed> > seedToken;

  std::string theAlias;

  void setAlias(std::string alias) {
    alias.erase(alias.size() - 1, alias.size());
    theAlias = alias;
  }
};

#endif
