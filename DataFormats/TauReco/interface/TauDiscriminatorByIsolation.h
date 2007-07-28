#ifndef TauReco_TauDiscriminatorByIsolation_h
#define TauReco_TauDiscriminatorByIsolation_h
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TauReco/interface/Tau.h"

#include <vector>

namespace reco {
  typedef edm::AssociationVector<TauRefProd,std::vector<double> > TauDiscriminatorByIsolation;
}
#endif

