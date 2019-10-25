#ifndef ElectronClassification_H
#define ElectronClassification_H

//===================================================================
// Author: Federico Ferri - INFN Milano, Bicocca university
// 12/2005
// See GsfElectron::Classification
//===================================================================

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

namespace electronAlgos {
  reco::GsfElectron::Classification classify(reco::GsfElectron const&);
}

#endif
