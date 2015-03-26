#ifndef RecoMuon_MuonSeedGenerator_MuonSeedMerger_H
#define RecoMuon_MuonSeedGenerator_MuonSeedMerger_H

/** \class MuonSeedMerger
 *  Module to merge two or more muon seed collections.
 *  Currently it does not contain any seed cleaner, so the number of ghosts seed
 *  can be high.
 *  This is still a preliminary implementation.
 *
 *  \author R. Bellan - CERN <riccardo.bellan@cern.ch>
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
// Data Formats 
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/View.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonSeedMerger : public edm::stream::EDProducer<> {
public:
  /// Constructor
  MuonSeedMerger(const edm::ParameterSet&);

  /// Destructor
  virtual ~MuonSeedMerger();

  // Operations

  virtual void produce(edm::Event&, const edm::EventSetup&) override;

protected:

private:
  std::vector<edm::InputTag> theSeedCollectionLabels;
  std::vector<edm::EDGetTokenT<edm::View<TrajectorySeed> > > seedTokens;

};
#endif

