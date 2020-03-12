#ifndef ElectronClassification_H
#define ElectronClassification_H

//===================================================================
// Author: Federico Ferri - INFN Milano, Bicocca university
// 12/2005
// See GsfElectron::Classification
//===================================================================

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

namespace egamma {
  reco::GsfElectron::Classification classifyElectron(reco::GsfElectron const&);
}

#endif
