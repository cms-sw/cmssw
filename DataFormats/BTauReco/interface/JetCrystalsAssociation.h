#ifndef BTauReco_JetCrystalsAssociation_h
#define BTauReco_JetCrystalsAssociation_h
// \class JetCrystalsAssociation
//
// \short association of Ecal Crystals to jet
//
//

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/LorentzVectorFwd.h"
#include <vector>

namespace reco {
  typedef math::PtEtaPhiELorentzVector           EMLorentzVector;
  typedef math::PtEtaPhiELorentzVectorCollection EMLorentzVectorCollection;
  typedef math::PtEtaPhiELorentzVectorRef        EMLorentzVectorRef;
  typedef math::PtEtaPhiELorentzVectorRefProd    EMLorentzVectorRefProd;
  typedef math::PtEtaPhiELorentzVectorRefVector  EMLorentzVectorRefVector;
/*
  typedef
    std::pair<edm::RefToBase<Jet>, EMLorentzVectorRefVector> JetCrystalsAssociation;
*/
  struct JetCrystalsAssociation :
    public std::pair<edm::RefToBase<Jet>, EMLorentzVectorRefVector>
  {
    typedef std::pair<edm::RefToBase<Jet>, EMLorentzVectorRefVector> base_class;

    JetCrystalsAssociation() :
      base_class() { }

    JetCrystalsAssociation(const base_class::first_type & first, const base_class::second_type & second) :
     base_class(first, second) { }

    JetCrystalsAssociation(const base_class & _pair) :
      base_class(_pair) { }

    JetCrystalsAssociation(const JetCrystalsAssociation & _pair) :
      base_class(_pair) { }
  };
  typedef
  std::vector<JetCrystalsAssociation> JetCrystalsAssociationCollection;

  typedef
  edm::Ref<JetCrystalsAssociationCollection> JetCrystalsAssociationRef;

  typedef
  edm::FwdRef<JetCrystalsAssociationCollection> JetCrystalsAssociationFwdRef;

  typedef
  edm::RefProd<JetCrystalsAssociationCollection> JetCrystalsAssociationRefProd;

  typedef
  edm::RefVector<JetCrystalsAssociationCollection> JetCrystalsAssociationRefVector;
}
#endif
