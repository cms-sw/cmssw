#ifndef DataFormats_JetMatching_JetMatchedPartons_h
#define DataFormats_JetMatching_JetMatchedPartons_h

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include "DataFormats/JetMatching/interface/MatchedPartons.h"
#include <vector>

namespace reco {

  typedef edm::AssociationVector<edm::RefToBaseProd<reco::Jet>, std::vector<reco::MatchedPartons> >
      JetMatchedPartonsCollectionBase;

  class JetMatchedPartonsCollection : public JetMatchedPartonsCollectionBase {
  public:
    JetMatchedPartonsCollection() : JetMatchedPartonsCollectionBase() {}

    JetMatchedPartonsCollection(const reco::CaloJetRefProd &ref)
        : JetMatchedPartonsCollectionBase(edm::RefToBaseProd<reco::Jet>(ref)) {}

    JetMatchedPartonsCollection(const JetMatchedPartonsCollectionBase &v) : JetMatchedPartonsCollectionBase(v) {}
  };

  typedef JetMatchedPartonsCollection::value_type JetMatchedPartons;

  typedef edm::Ref<JetMatchedPartonsCollection> JetMatchedPartonsRef;

  typedef edm::RefProd<JetMatchedPartonsCollection> JetMatchedPartonsRefProd;

  typedef edm::RefVector<JetMatchedPartonsCollection> JetMatchedPartonsRefVector;

}  // namespace reco

#endif  // DataFormats_JetMatching_JetMatchedPartons_h
