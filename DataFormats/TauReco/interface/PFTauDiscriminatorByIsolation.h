#ifndef DataFormats_TauReco_PFTauDiscriminatorByIsolation_h
#define DataFormats_TauReco_PFTauDiscriminatorByIsolation_h
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TauReco/interface/PFTau.h"

#include <vector>

namespace reco {
  typedef edm::AssociationVector<PFTauRefProd,std::vector<double> > PFTauDiscriminatorByIsolation;
}
#endif

