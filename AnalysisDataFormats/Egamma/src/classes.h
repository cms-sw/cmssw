#include "DataFormats/Common/interface/Wrapper.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronID.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronIDAssociation.h"
#include "DataFormats/Common/interface/RefToBase.h"

namespace
{
   struct dictionary
   {
      reco::ElectronIDCollection c1;
      edm::Wrapper<reco::ElectronIDCollection> w1;
      edm::Ref<reco::ElectronIDCollection> r1;
      edm::RefProd<reco::ElectronIDCollection> rp1;
      edm::RefVector<reco::ElectronIDCollection> rv1;

      reco::ElectronIDAssociationCollection c2;
      edm::Wrapper<reco::ElectronIDAssociationCollection> w2;
      reco::ElectronIDAssociation va1;
      reco::ElectronIDAssociationRef vr1;
      reco::ElectronIDAssociationRefProd vrp1;
      reco::ElectronIDAssociationRefVector vrv1;
   };
}
