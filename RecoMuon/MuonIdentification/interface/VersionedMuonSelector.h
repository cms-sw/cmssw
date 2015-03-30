#ifndef __RecoMuon_MuonIdentification_VersionedMuonIdProducer__
#define __RecoMuon_MuonIdentification_VersionedMuonIdProducer__

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"

#include "PhysicsTools/SelectorUtils/interface/VersionedSelector.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

typedef VersionedSelector<edm::Ptr<reco::Muon> > VersionedMuonSelector;

#endif
