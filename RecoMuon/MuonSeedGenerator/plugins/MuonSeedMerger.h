#ifndef RecoMuon_MuonSeedGenerator_MuonSeedMerger_H
#define RecoMuon_MuonSeedGenerator_MuonSeedMerger_H

/** \class MuonSeedMerger
 *  Module to merge two or more muon seed collections.
 *  Currently it does not contain any seed cleaner, so the number of ghosts seed
 *  can be high.
 *  This is still a preliminary implementation.
 *
 *  $Date: 2010/02/16 17:08:40 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - CERN <riccardo.bellan@cern.ch>
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonSeedMerger : public edm::EDProducer {
public:
  /// Constructor
  MuonSeedMerger(const edm::ParameterSet&);

  /// Destructor
  virtual ~MuonSeedMerger();

  // Operations

  virtual void produce(edm::Event&, const edm::EventSetup&);

protected:

private:
  std::vector<edm::InputTag> theSeedCollectionLabels;
};
#endif

