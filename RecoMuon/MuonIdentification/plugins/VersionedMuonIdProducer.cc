// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"

#include "PhysicsTools/SelectorUtils/interface/VersionedIdProducer.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

typedef VersionedIdProducer<edm::Ptr<reco::Muon> > VersionedMuonIdProducer;

//define this as a plug-in
DEFINE_FWK_MODULE(VersionedMuonIdProducer);
