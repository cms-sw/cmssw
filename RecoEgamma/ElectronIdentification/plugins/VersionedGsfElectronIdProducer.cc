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

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

typedef VersionedIdProducer<reco::GsfElectronPtr> VersionedGsfElectronIdProducer;

//define this as a plug-in
DEFINE_FWK_MODULE(VersionedGsfElectronIdProducer);
