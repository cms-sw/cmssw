#include "ExtractorFromDeposits.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;
using reco::isodeposit::Direction;

ExtractorFromDeposits::ExtractorFromDeposits( const ParameterSet& par ) :
  theCollectionTag(par.getParameter<edm::InputTag>("IsolationCollectionTag"))
{ }

void ExtractorFromDeposits::fillVetos (const edm::Event & ev, 
    const edm::EventSetup & evSetup, const reco::TrackCollection & muons) 
{ }

IsoDeposit ExtractorFromDeposits::deposit(const Event & event, 
    const EventSetup & eventSetup, const Track & muon) const
{ 
  static std::string metname = "RecoMuon|ExtractorFromDeposits";
  Handle<reco::IsoDepositMap> depMap;
  event.getByLabel(theCollectionTag, depMap);

  LogWarning(metname)<<"Call this method only if the original muon track collection is lost";

  // double vtx_z = muon.vz();
  reco::isodeposit::Direction muonDir(muon.eta(), muon.phi());

  typedef reco::IsoDepositMap::const_iterator iterator_i;
  typedef reco::IsoDepositMap::container::const_iterator iterator_ii;
  iterator_i depI = depMap->begin();
  iterator_i depIEnd = depMap->end();
  for (; depI != depIEnd; ++depI){
    iterator_ii depII = depI.begin();
    iterator_ii depIIEnd = depI.end();
    for (; depII != depIIEnd; ++depII){
      reco::isodeposit::Direction depDir(depII->direction());
      if (muonDir.deltaR(depDir) < 1.e-6) return *depII;
    }
  }

  return IsoDeposit();
}

IsoDeposit ExtractorFromDeposits::deposit(const Event & event, 
    const EventSetup & eventSetup, const TrackRef & muon) const
{ 
  static std::string metname = "RecoMuon|ExtractorFromDeposits";
  Handle<reco::IsoDepositMap> depMap;
  event.getByLabel(theCollectionTag, depMap);

  return (*depMap)[muon];
}
