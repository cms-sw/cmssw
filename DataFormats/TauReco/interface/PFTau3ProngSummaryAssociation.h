#ifndef DataFormats_TauReco_PPFTau3ProngSummaryAssociation_h
#define DataFormats_TauReco_PPFTau3ProngSummaryAssociation_h

#include "DataFormats/TauReco/interface/PFTau3ProngSummary.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

namespace reco {
  // PPFTau3ProngSummary
  typedef edm::AssociationVector<PFTauRefProd, std::vector<reco::PFTau3ProngSummaryRef> > PFTau3ProngSumAssociation;
  typedef PFTau3ProngSumAssociation::value_type PFTau3ProngSumAssociationVT;  
  typedef edm::Ref<PFTau3ProngSumAssociation> PFTau3ProngSumAssociationRef;  
  typedef edm::RefProd<PFTau3ProngSumAssociation> PFTau3ProngSumAssociationRefProd;  
  typedef edm::RefVector<PFTau3ProngSumAssociation> PFTau3ProngSumAssociationRefVector; 
}

#endif
