#ifndef DataFormats_TauReco_PFTauDecayModeAssociation_h
#define DataFormats_TauReco_PFTauDecayModeAssociation_h

#include "DataFormats/TauReco/interface/PFTauDecayMode.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TauReco/interface/PFTau.h"

namespace reco {
  //used for matching Truth to Reco PFTauDecayModeCollections
  //typedef edm::Association<reco::PFTauDecayModeCollection> PFTauDecayModeMatchMap;
  typedef edm::Association<reco::PFTauCollection> PFTauDecayModeMatchMap;

  //actual product of PFTauDecayModeProducer, maps PFTauDecayModes to their associated PFTaus
  typedef edm::AssociationVector<PFTauRefProd, reco::PFTauDecayModeCollection> PFTauDecayModeAssociation;
  typedef PFTauDecayModeAssociation::value_type PFTauDecayModeAssociationVT;
  typedef edm::Ref<PFTauDecayModeAssociation> PFTauDecayModeAssociationRef;
  typedef edm::RefProd<PFTauDecayModeAssociation> PFTauDecayModeAssociationRefProd;
  typedef edm::RefVector<PFTauDecayModeAssociation> PFTauDecayModeAssociationRefVector;
}  // namespace reco

#endif
