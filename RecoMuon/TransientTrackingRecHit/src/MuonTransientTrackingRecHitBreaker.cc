#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBreaker.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

TransientTrackingRecHit::ConstRecHitContainer MuonTransientTrackingRecHitBreaker::breakInSubRecHits(
    TransientTrackingRecHit::ConstRecHitPointer muonRecHit, int granularity) {
  const std::string metname = "Muon|RecoMuon|MuonTransientTrackingRecHitBreaker";

  TransientTrackingRecHit::ConstRecHitContainer recHitsForFit;

  int subDet = muonRecHit->geographicalId().subdetId();

  switch (granularity) {
    case 0: {
      // Asking for 4D segments for the CSC/DT and a point for the RPC
      recHitsForFit.push_back(muonRecHit);
      break;
    }
    case 1: {
      if (subDet == MuonSubdetId::DT || subDet == MuonSubdetId::CSC)
        // measurement->recHit() returns a 4D segment, then
        // DT case: asking for 2D segments.
        // CSC case: asking for 2D points.
        recHitsForFit = muonRecHit->transientHits();

      else if (subDet == MuonSubdetId::RPC)
        recHitsForFit.push_back(muonRecHit);

      break;
    }

    case 2: {
      if (subDet == MuonSubdetId::DT) {
        // Asking for 2D segments. measurement->recHit() returns a 4D segment
        TransientTrackingRecHit::ConstRecHitContainer segments2D = muonRecHit->transientHits();

        // loop over segment
        for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator segment = segments2D.begin();
             segment != segments2D.end();
             ++segment) {
          // asking for 1D Rec Hit
          TransientTrackingRecHit::ConstRecHitContainer rechit1D = (**segment).transientHits();

          // load them into the recHitsForFit container
          copy(rechit1D.begin(), rechit1D.end(), back_inserter(recHitsForFit));
        }
      }

      else if (subDet == MuonSubdetId::RPC)
        recHitsForFit.push_back(muonRecHit);

      else if (subDet == MuonSubdetId::CSC)
        // Asking for 2D points. measurement->recHit() returns a 4D segment
        recHitsForFit = (*muonRecHit).transientHits();

      break;
    }

    default: {
      throw cms::Exception(metname) << "Wrong granularity chosen!"
                                    << "it will be set to 0";
      break;
    }
  }

  return recHitsForFit;
}
