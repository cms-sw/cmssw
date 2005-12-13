#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/Merger.h"
#include "PhysicsTools/Utilities/interface/CopyPolicy.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EGammaReco/interface/Electron.h"
#include "DataFormats/EGammaReco/interface/Photon.h"

typedef Merger<reco::TrackCollection,    CopyPolicy<reco::Track> >    TrackMerger;
typedef Merger<reco::MuonCollection,     CopyPolicy<reco::Muon> >     MuonMerger;
typedef Merger<reco::ElectronCollection, CopyPolicy<reco::Electron> > ElectronMerger;
typedef Merger<reco::PhotonCollection,   CopyPolicy<reco::Photon> >   PhotonMerger;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( TrackMerger );
DEFINE_ANOTHER_FWK_MODULE( MuonMerger );
DEFINE_ANOTHER_FWK_MODULE( ElectronMerger );
DEFINE_ANOTHER_FWK_MODULE( PhotonMerger );

