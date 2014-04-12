#include "DPGAnalysis/Skims/interface/TriggerMatchProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
typedef TriggerMatchProducer<reco::IsolatedPixelTrackCandidate> trgMatchIsolatedPixelTrackCandidateProducer;
DEFINE_FWK_MODULE( trgMatchIsolatedPixelTrackCandidateProducer );
