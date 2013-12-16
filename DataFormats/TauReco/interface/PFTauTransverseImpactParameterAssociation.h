#ifndef DataFormats_TauReco_PFTauTransverseImpactParameterAssociation_h
#define DataFormats_TauReco_PFTauTransverseImpactParameterAssociation_h

#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameter.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterFwd.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

namespace reco {
  // PFTauTransverseImpactParameter
  typedef edm::AssociationVector<PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef> > PFTauTIPAssociation;
  typedef PFTauTIPAssociation::value_type PFTauTIPAssociationVT;  
  typedef edm::Ref<PFTauTIPAssociation> PFTauTIPAssociationRef;  
  typedef edm::RefProd<PFTauTIPAssociation> PFTauTIPAssociationRefProd;  
  typedef edm::RefVector<PFTauTIPAssociation> PFTauTIPAssociationRefVector; 
  // std::vector<reco::vertex>
  typedef edm::AssociationVector<PFTauRefProd, std::vector<reco::VertexRef> > PFTauVertexAssociation;
  typedef PFTauVertexAssociation::value_type PFTauVertexAssociationVT;
  typedef edm::Ref<PFTauVertexAssociation> PFTauVertexAssociationRef;
  typedef edm::RefProd<PFTauVertexAssociation> PFTauVertexAssociationRefProd;
  typedef edm::RefVector<PFTauVertexAssociation> PFTauVertexAssociationRefVector;
  // std::vector<std::vector<reco::Vertex> >
  typedef edm::AssociationVector<PFTauRefProd,  std::vector<std::vector<reco::VertexRef> > > PFTauVertexVAssociation;
  typedef PFTauVertexVAssociation::value_type PFTauVertexVAssociationVT;
  typedef edm::Ref<PFTauVertexVAssociation> PFTauVertexVAssociationRef;
  typedef edm::RefProd<PFTauVertexVAssociation> PFTauVertexVAssociationRefProd;
  typedef edm::RefVector<PFTauVertexVAssociation> PFTauVertexVAssociationRefVector;

}

#endif
