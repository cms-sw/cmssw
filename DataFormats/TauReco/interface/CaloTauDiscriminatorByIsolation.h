#ifndef DataFormats_TauReco_CaloTauDiscriminatorByIsolation_h
#define DataFormats_TauReco_CaloTauDiscriminatorByIsolation_h
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TauReco/interface/CaloTau.h"

#include <vector>

namespace reco {
  typedef edm::AssociationVector<CaloTauRefProd,std::vector<double> > CaloTauDiscriminatorByIsolation;
}
#endif

