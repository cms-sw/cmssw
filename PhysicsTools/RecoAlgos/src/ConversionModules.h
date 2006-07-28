#ifndef RecoAlgos_ConversionModules_h
#define RecoAlgos_ConversionModules_h

#include "PhysicsTools/CandAlgos/interface/RecoToCandCollectionConverter.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

namespace reco {
  namespace modules {
    /// Converter from reco::ElectronCollection to CandidateCollection 
    typedef RecoToCandCollectionConverter<reco::ElectronCollection> ElectronToCandCollectionConverter;
  }
}

#endif
